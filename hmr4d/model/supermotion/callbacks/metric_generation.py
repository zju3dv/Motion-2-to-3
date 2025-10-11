##############
# Most of them are from https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/modeltype/mld.py
##############
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.metric_utils import ListAggregator
from hmr4d.utils.pylogger import Log
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase
from hmr4d.dataset.hml3d.hml3d_utils import resample_motion_fps
from hydra.utils import instantiate
from hmr4d.utils.hml3d.metric import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_diversity_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_multimodality_np,
)
import hmr4d.network.evaluator.t2m_motionenc as t2m_motionenc
import hmr4d.network.evaluator.t2m_textenc as t2m_textenc
from hmr4d.network.evaluator.word_vectorizer import POS_enumerator


def is_target_task(batch):
    task = check_equal_get_one(batch["task"], "task")
    return task == "GEN"


class MetricT2M(pl.Callback):
    def __init__(
        self,
        endecoder_opt,
        top_k=3,
        R_size=32,
        diversity_times=300,
        is_export=False,
        exp_name=None,
        data_name=None,
    ):
        super().__init__()
        self.endecoder_opt = endecoder_opt
        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.is_export = is_export
        self.exp_name = exp_name
        self.data_name = data_name
        self.text_embeddings = ListAggregator()
        self.gen_motion_embeddings = ListAggregator()
        self.gt_motion_embeddings = ListAggregator()
        self._get_t2m_evaluator()

        self.data_endecoder: EnDecoderBase = instantiate(endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Send models to GPU
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    def _get_t2m_evaluator(self):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        ######
        # OPT is from https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/networks/evaluator_wrapper.py
        ######
        opt = {
            "dim_word": 300,
            "max_motion_length": 196,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 263,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": "./inputs/checkpoints/t2m",
            "unit_length": 4,
        }
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=opt["dim_word"],
            pos_size=opt["dim_pos_ohot"],
            hidden_size=opt["dim_text_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=opt["dim_pose"] - 4,
            hidden_size=opt["dim_movement_enc_hidden"],
            output_size=opt["dim_movement_latent"],
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=opt["dim_movement_latent"],
            hidden_size=opt["dim_motion_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )
        # load pretrianed
        t2m_checkpoint = torch.load(
            os.path.join(opt["checkpoints_dir"], "text_mot_match/model/finest.tar"),
            map_location="cpu",
        )
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        The behaviour is the same for val/test/predict
        """
        if not is_target_task(batch):
            return

        B = batch["length"].shape[0]
        length = batch["length"]
        ori_length = batch["ori_length"]
        device = batch["length"].device

        pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, L, 22, 3)
        gt_ayfz_motion = batch["gt_ayfz_motion"]  # (B, L, 22, 3)

        word_embs = batch["word_embs"]
        pos_onehot = batch["pos_onehot"]
        text_length = batch["text_len"]

        # Check if pred_motion has nan, if so, set it to zero and print a warning
        if torch.isnan(pred_ayfz_motion).any():
            nan_mask = torch.isnan(pred_ayfz_motion)
            num_nan_item = (nan_mask.view(B, -1).any(dim=-1)).sum()
            Log.warn(f"{num_nan_item} pred_motion has nan")
            pred_ayfz_motion[nan_mask] = 0

        # resample to original fps (20 fps)
        pred_ayfz_motion_ = torch.zeros((B, 200, 22, 3), device=device)
        gt_ayfz_motion_ = torch.zeros((B, 200, 22, 3), device=device)
        for i in range(B):
            ori_len = ori_length[i]
            now_len = length[i]
            pred_ayfz_motion_[i, :ori_len] = resample_motion_fps(pred_ayfz_motion[i, :now_len], ori_len)
            gt_ayfz_motion_[i, :ori_len] = resample_motion_fps(gt_ayfz_motion[i, :now_len], ori_len)

        # normalize
        pred_ayfz_motion_vec = self.encoder_motion3d(pred_ayfz_motion_, ori_length)
        gt_ayfz_motion_vec = self.encoder_motion3d(gt_ayfz_motion_, ori_length)
        # (B, C, L) -> (B, L, C)
        pred_ayfz_motion_vec = pred_ayfz_motion_vec.transpose(1, 2)
        gt_ayfz_motion_vec = gt_ayfz_motion_vec.transpose(1, 2)

        # t2m motion encoder
        m_lens = torch.div(ori_length, 4, rounding_mode="floor")

        # motion length should be sorted in decreasing order for RNN batch forward
        align_m_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        pred_ayfz_motion_vec = pred_ayfz_motion_vec[align_m_idx]
        gt_ayfz_motion_vec = gt_ayfz_motion_vec[align_m_idx]
        m_lens = m_lens[align_m_idx]

        motion_mov = self.t2m_moveencoder(pred_ayfz_motion_vec[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        gt_motion_mov = self.t2m_moveencoder(gt_ayfz_motion_vec[..., :-4]).detach()
        gt_motion_emb = self.t2m_motionencoder(gt_motion_mov, m_lens)

        # t2m text encoder
        # text length should be sorted in decreasing order for RNN batch forward
        align_t_idx = np.argsort(text_length.data.tolist())[::-1].copy()
        word_embs = word_embs[align_t_idx]
        pos_onehot = pos_onehot[align_t_idx]
        text_length = text_length[align_t_idx]
        text_emb = self.t2m_textencoder(word_embs, pos_onehot, text_length)
        # text order convert to motion order
        inverse_align_t_idx = np.argsort(align_t_idx)
        text_emb = text_emb[inverse_align_t_idx][align_m_idx]

        self.text_embeddings.update(text_emb)
        self.gen_motion_embeddings.update(motion_emb)
        self.gt_motion_embeddings.update(gt_motion_emb)

    def on_predict_epoch_start(self, trainer, pl_module):
        self.text_embeddings.reset()
        self.gen_motion_embeddings.reset()
        self.gt_motion_embeddings.reset()

        # NOTE: not sure whether this way is beautiful
        self.t2m_textencoder = self.t2m_textencoder.to(pl_module.device)
        self.t2m_moveencoder = self.t2m_moveencoder.to(pl_module.device)
        self.t2m_motionencoder = self.t2m_motionencoder.to(pl_module.device)
        self.data_endecoder = self.data_endecoder.to(pl_module.device)

        if self.exp_name is None:
            if hasattr(pl_module, "exp_name"):
                self.exp_name = pl_module.exp_name
            else:
                self.exp_name = "Unnamed_Experiment"
        if self.data_name is None:
            if hasattr(pl_module, "data_name"):
                self.data_name = pl_module.data_name
            else:
                self.data_name = "Unknown_Data"
        self.seed = pl_module.seed

    # ================== Epoch Summary  ================== #
    @rank_zero_only
    def on_predict_epoch_end(self, trainer, pl_module):
        metrics = {
            "Matching_score": 0.0,
            "gt_Matching_score": 0.0,
            "Diversity": 0.0,
            "gt_Diversity": 0.0,
        }
        for i in range(self.top_k):
            metrics[f"R_precision_top_{str(i + 1)}"] = 0.0
            metrics[f"gt_R_precision_top_{str(i + 1)}"] = 0.0

        count_seq = self.text_embeddings.length()

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = self.text_embeddings.get_tensor()[shuffle_idx]
        all_genmotions = self.gen_motion_embeddings.get_tensor()[shuffle_idx]
        all_gtmotions = self.gt_motion_embeddings.get_tensor()[shuffle_idx]

        device = all_genmotions.device

        # Compute r-precision
        assert count_seq > self.R_size
        matching_score = 0.0
        top_k_mat = torch.zeros((self.top_k,), device=device)
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size : (i + 1) * self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            # print(dist_mat[:5])
            matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq >= self.R_size
        matching_score = 0.0
        top_k_mat = torch.zeros((self.top_k,), device=device)
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            # match score
            matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["gt_Matching_score"] = matching_score / R_count
        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.detach().cpu().numpy()
        all_gtmotions = all_gtmotions.detach().cpu().numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_stat = torch.load("./hmr4d/utils/hml3d/gt_stat.pth")
        gt_ori_mu, gt_ori_cov = gt_stat["mu"], gt_stat["cov"]
        metrics["FID_our"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        metrics["FID_original"] = calculate_frechet_distance_np(gt_ori_mu, gt_ori_cov, mu, cov)

        # Compute diversity
        if count_seq > self.diversity_times:
            diversity_times = self.diversity_times
        else:
            diversity_times = count_seq
            Log.warn(
                f"Generation metric - Diversity required {self.diversity_times} sequences, "
                f"but only uses {diversity_times} sequences to calculate!"
            )
        metrics["Diversity"] = calculate_diversity_np(all_genmotions, diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions, diversity_times)

        # log to stdout
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            Log.info(f"{k}: {v:.3f}")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                pl_module.logger.log_metrics({f"val_metric/{k}": v}, step=cur_epoch)

        self.text_embeddings.reset()
        self.gen_motion_embeddings.reset()
        self.gt_motion_embeddings.reset()

        self.t2m_textencoder = self.t2m_textencoder.cpu()
        self.t2m_moveencoder = self.t2m_moveencoder.cpu()
        self.t2m_motionencoder = self.t2m_motionencoder.cpu()
        self.data_endecoder = self.data_endecoder.cpu()

        if self.is_export:
            save_path = os.path.join("./outputs", self.data_name, self.exp_name, "evaluation")
            os.makedirs(save_path, exist_ok=True)
            i = 0
            pt_name = f"metric_t2m_{self.seed}_0.pt"
            while os.path.exists(os.path.join(save_path, pt_name)):
                i += 1
                pt_name = f"metric_t2m_{self.seed}_{i}.pt"
            torch.save(metrics, os.path.join(save_path, pt_name))
            Log.info(f"Save text2motion metrics in {os.path.join(save_path, pt_name)}")


class MetricMM(pl.Callback):
    def __init__(self, endecoder_opt, mm_num_times=10, is_export=True, exp_name=None, data_name=None):
        super().__init__()
        self.endecoder_opt = endecoder_opt
        self.mm_num_times = mm_num_times
        self.is_export = is_export
        self.exp_name = exp_name
        self.data_name = data_name
        self.gen_motion_embeddings = ListAggregator()
        self._get_mm_evaluator()

        self.data_endecoder: EnDecoderBase = instantiate(endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Send models to GPU
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    def _get_mm_evaluator(self):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        ######
        # OPT is from https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/networks/evaluator_wrapper.py
        ######
        opt = {
            "dim_word": 300,
            "max_motion_length": 196,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 263,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": "./inputs/checkpoints/t2m",
            "unit_length": 4,
        }
        # init module
        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=opt["dim_pose"] - 4,
            hidden_size=opt["dim_movement_enc_hidden"],
            output_size=opt["dim_movement_latent"],
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=opt["dim_movement_latent"],
            hidden_size=opt["dim_motion_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )
        # load pretrianed
        t2m_checkpoint = torch.load(
            os.path.join(opt["checkpoints_dir"], "text_mot_match/model/finest.tar"),
            map_location="cpu",
        )
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        The behaviour is the same for val/test/predict
        """
        if not is_target_task(batch):
            return

        B = batch["length"].shape[0]
        length = batch["length"]
        ori_length = batch["ori_length"]
        device = batch["length"].device

        pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, L, 22, 3)

        # Check if pred_motion has nan, if so, set it to zero and print a warning
        if torch.isnan(pred_ayfz_motion).any():
            nan_mask = torch.isnan(pred_ayfz_motion)
            num_nan_item = (nan_mask.view(B, -1).any(dim=-1)).sum()
            Log.warn(f"{num_nan_item} pred_motion has nan")
            pred_ayfz_motion[nan_mask] = 0

        # resample to original fps (20 fps)
        pred_ayfz_motion_ = torch.zeros((B, 200, 22, 3), device=device)
        for i in range(B):
            ori_len = ori_length[i]
            now_len = length[i]
            pred_ayfz_motion_[i, :ori_len] = resample_motion_fps(pred_ayfz_motion[i, :now_len], ori_len)

        # normalize
        pred_ayfz_motion_vec = self.encoder_motion3d(pred_ayfz_motion_, ori_length)
        # (B, C, L) -> (B, L, C)
        pred_ayfz_motion_vec = pred_ayfz_motion_vec.transpose(1, 2)

        # t2m motion encoder
        m_lens = torch.div(ori_length, 4, rounding_mode="floor")

        # motion length should be sorted in decreasing order for RNN batch forward
        align_m_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        pred_ayfz_motion_vec = pred_ayfz_motion_vec[align_m_idx]
        m_lens = m_lens[align_m_idx]

        motion_mov = self.t2m_moveencoder(pred_ayfz_motion_vec[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)  # (B, C)

        self.gen_motion_embeddings.update(motion_emb[None])  # (1, B, C)

    def on_predict_epoch_start(self, trainer, pl_module):
        self.gen_motion_embeddings.reset()

        # NOTE: not sure whether this way is beautiful
        self.t2m_moveencoder = self.t2m_moveencoder.to(pl_module.device)
        self.t2m_motionencoder = self.t2m_motionencoder.to(pl_module.device)
        self.data_endecoder = self.data_endecoder.to(pl_module.device)

        if self.exp_name is None:
            if hasattr(pl_module, "exp_name"):
                self.exp_name = pl_module.exp_name
            else:
                self.exp_name = "Unnamed_Experiment"
        if self.data_name is None:
            if hasattr(pl_module, "data_name"):
                self.data_name = pl_module.data_name
            else:
                self.data_name = "Unknown_Data"
        self.seed = pl_module.seed

    # ================== Epoch Summary  ================== #
    @rank_zero_only
    def on_predict_epoch_end(self, trainer, pl_module):
        metrics = {
            "MultiModality": 0.0,
        }
        # cat all embeddings
        all_mm_motions = self.gen_motion_embeddings.get_tensor()  # (B, 30, C)
        all_mm_motions = all_mm_motions.detach().cpu().numpy()

        metrics["MultiModality"] = calculate_multimodality_np(all_mm_motions, self.mm_num_times)

        # log to stdout
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            Log.info(f"{k}: {v:.3f}")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                pl_module.logger.log_metrics({f"val_metric/{k}": v}, step=cur_epoch)

        self.gen_motion_embeddings.reset()

        self.t2m_moveencoder = self.t2m_moveencoder.cpu()
        self.t2m_motionencoder = self.t2m_motionencoder.cpu()
        self.data_endecoder = self.data_endecoder.cpu()

        if self.is_export:
            save_path = os.path.join("./outputs", self.data_name, self.exp_name, "evaluation")
            os.makedirs(save_path, exist_ok=True)
            i = 0
            pt_name = f"metric_mm_{self.seed}_0.pt"
            while os.path.exists(os.path.join(save_path, pt_name)):
                i += 1
                pt_name = f"metric_mm_{self.seed}_{i}.pt"
            torch.save(metrics, os.path.join(save_path, pt_name))
            Log.info(f"Save multimodality metrics in {os.path.join(save_path, pt_name)}")
