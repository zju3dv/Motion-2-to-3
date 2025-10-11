import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.configs import MainStore, builds

from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.pylogger import Log
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
)
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.smplx_utils import make_smplx
from einops import einsum, rearrange

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static
from hmr4d.utils.geo.hmr_cam import estimate_focal_length
import imageio
import decord
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2

from smplx.joint_names import JOINT_NAMES


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }

        self.perjoint_metrics = False
        if self.perjoint_metrics:
            body_joint_names = JOINT_NAMES[:22] + ["left_hand", "right_hand"]
            self.body_joint_names = body_joint_names
            self.perjoint_metric_aggregator = {
                "mpjpe": {k: {} for k in body_joint_names},
            }
            self.perjoint_obs_metric_aggregator = {
                "mpjpe": {k: {} for k in body_joint_names},
            }

        # SMPL
        # self.smplh_model = {
        #     "male": make_smplx("rich-smplh", gender="male").cuda(),
        #     "female": make_smplx("rich-smplh", gender="female").cuda(),
        #     "neutral": make_smplx("rich-smplh", gender="neutral").cuda(),
        # }
        self.smplx_model = {
            "male": make_smplx("rich-smplx", gender="male"),
            "female": make_smplx("rich-smplx", gender="female"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
        }
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smpl = make_smplx("rich-smplh").bm.faces
        self.faces_smplx = self.smplx_model["neutral"].bm.faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        if not is_target_task(batch):
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        assert batch["B"] == 1, "The evaluation is performed in a sequence-to-sequence manner."
        vid = batch["meta"][0]["vid"]
        device = batch["length"].device
        seq_length = batch["length"][0].item()
        gender = batch["gender"][0]
        T_w2ay = batch["T_w2ay"][0]
        T_w2c = batch["T_w2c"][0]

        # Count on sequence

        # assign gt betas to check pose error, BUG: not right as RICH employs gender model
        # outputs["pred_smpl_params_incam"]["betas"] = batch["gt_smpl_params"]["betas"][0]

        # Parse prediction
        # 1. cam
        pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        offset = pred_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
        pred_cr_j3d = pred_c_j3d - offset
        pred_cr_verts = pred_c_verts - offset

        # 2. ay
        pred_smpl_params_global = outputs["pred_smpl_params_global"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_global)
        pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")

        # Groundtruth (rich use gender)
        # 1. cam
        target_w_params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
        target_w_output = self.smplx_model[gender](**target_w_params)
        target_w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices])
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
        offset = target_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
        target_cr_j3d = target_c_j3d - offset
        target_cr_verts = target_c_verts - offset

        # 2. ay
        target_ay_verts = apply_T_on_points(target_w_verts, T_w2ay)
        target_ay_j3d = torch.matmul(self.J_regressor, target_ay_verts)

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_cr_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_cr_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        if self.perjoint_metrics:
            camcoord_perjoint_metrics = compute_camcoord_perjoint_metrics(batch_eval)
            for k in camcoord_perjoint_metrics:
                for j in range(camcoord_perjoint_metrics[k].shape[1]):
                    k_ = self.body_joint_names[j]
                    self.perjoint_metric_aggregator[k][k_][vid] = as_np_array(camcoord_perjoint_metrics[k][:, j])

        batch_eval = {
            "pred_j3d_glob": pred_ay_j3d,
            "target_j3d_glob": target_ay_j3d,
            "pred_verts_glob": pred_ay_verts,
            "target_verts_glob": target_ay_verts,
        }
        global_metrics = compute_global_metrics(batch_eval)
        for k in global_metrics:
            self.metric_aggregator[k][vid] = as_np_array(global_metrics[k])

        if self.perjoint_metrics:
            # Parse obs
            obs_params = {k: v[0] for k, v in batch["obs_smpl_params"].items()}  # remove fake bid
            obs_params["betas"] = obs_params["betas"][:, :10]  # The result should be similar
            smpl_out = self.smplx_model["neutral"](**obs_params)
            obs_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
            obs_c_j3d = einsum(self.J_regressor, obs_c_verts, "j v, l v i -> l j i")
            offset = obs_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
            obs_cr_j3d = obs_c_j3d - offset
            obs_cr_verts = obs_c_verts - offset

            # Metric of observation
            batch_eval = {
                "pred_j3d": obs_cr_j3d,
                "target_j3d": target_c_j3d,
                "pred_verts": obs_cr_verts,
                "target_verts": target_c_verts,
            }
            camcoord_metrics = compute_camcoord_metrics(batch_eval)
            for k in camcoord_metrics:
                camcoord_metrics[k] = as_np_array(camcoord_metrics[k])

            camcoord_perjoint_metrics = compute_camcoord_perjoint_metrics(batch_eval)
            for k in camcoord_perjoint_metrics:
                for j in range(camcoord_perjoint_metrics[k].shape[1]):
                    k_ = self.body_joint_names[j]
                    self.perjoint_obs_metric_aggregator[k][k_][vid] = as_np_array(camcoord_perjoint_metrics[k][:, j])

        # DEBUG visualize
        if False:
            # Print per-sequence error
            Log.info(
                f"seq {vid} metrics:\n"
                + "\n".join(
                    f"{k}: {self.metric_aggregator[k][vid].mean():.1f} (obs:{camcoord_metrics[k].mean():.1f})"
                    for k in camcoord_metrics.keys()
                )
                + "\n------\n"
            )
            if self.perjoint_metrics:
                Log.info(
                    f"\n".join(
                        f"{k}-{j}: {self.perjoint_metric_aggregator[k][j][vid].mean():.1f} (obs:{self.perjoint_obs_metric_aggregator[k][j][vid].mean():.1f})"
                        for j in self.body_joint_names
                        for k in self.perjoint_obs_metric_aggregator.keys()
                    )
                    + "\n------"
                )

            # -- metric -- #
            pred_mpjpe = self.metric_aggregator["mpjpe"][vid].mean()
            obs_mpjpe = camcoord_metrics["mpjpe"].mean()

            # -- render mesh -- #
            vertices_gt = target_c_verts
            vertices_cr_gt = target_cr_verts + target_cr_verts.new([0, 0, 3.0])  # move forward +z
            vertices_pred = pred_c_verts
            vertices_cr_obs = obs_cr_verts + obs_cr_verts.new([0, 0, 3.0])  # move forward +z
            vertices_cr_pred = pred_cr_verts + pred_cr_verts.new([0, 0, 3.0])  # move forward +z

            # -- rendering code -- #
            vname = batch["meta_render"][0]["name"]
            K = batch["meta_render"][0]["K"]
            width, height = batch["meta_render"][0]["width_height"]
            faces = self.faces_smpl

            renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
            out_fn = f"outputs/dump_render/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
            writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)

            # imgs
            video_path = batch["meta_render"][0]["video_path"]
            frame_id = batch["meta_render"][0]["frame_id"].cpu().numpy()
            vr = decord.VideoReader(video_path)
            images = vr.get_batch(list(frame_id)).numpy()  # (F, H/4, W/4, 3), uint8, numpy

            for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
                img_overlay_gt = renderer.render_mesh(vertices_gt[i].cuda(), images[i], [39, 194, 128])
                if batch["meta_render"][0].get("bbx_xys", None) is not None:  # draw bbox lines
                    bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                    img_overlay_gt = cv2.rectangle(img_overlay_gt, lu_point, rd_point, (255, 178, 102), 2)

                img_overlay_pred = renderer.render_mesh(vertices_pred[i].cuda(), images[i])
                # img_overlay_pred = renderer.render_mesh(vertices_pred[i].cuda(), np.zeros_like(images[i]))
                img = np.concatenate([img_overlay_gt, img_overlay_pred], axis=0)

                ####### overlay gt cr first, then overlay pred cr with error color ########
                # overlay gt cr first with blue color
                black_overlay_obs = renderer.render_mesh(
                    vertices_cr_gt[i].cuda(), np.zeros_like(images[i]), colors=[39, 194, 128]
                )
                black_overlay_pred = renderer.render_mesh(
                    vertices_cr_gt[i].cuda(), np.zeros_like(images[i]), colors=[39, 194, 128]
                )

                # get error color
                obs_error = (vertices_cr_gt[i] - vertices_cr_obs[i]).norm(dim=-1)
                pred_error = (vertices_cr_gt[i] - vertices_cr_pred[i]).norm(dim=-1)
                max_error = max(obs_error.max(), pred_error.max())
                obs_error_color = torch.stack(
                    [obs_error / max_error, torch.ones_like(obs_error) * 0.6, torch.ones_like(obs_error) * 0.6],
                    dim=-1,
                )
                obs_error_color = torch.clip(obs_error_color, 0, 1)
                pred_error_color = torch.stack(
                    [pred_error / max_error, torch.ones_like(pred_error) * 0.6, torch.ones_like(pred_error) * 0.6],
                    dim=-1,
                )
                pred_error_color = torch.clip(pred_error_color, 0, 1)

                # overlay cr with error color
                black_overlay_obs = renderer.render_mesh(
                    vertices_cr_obs[i].cuda(), black_overlay_obs, colors=obs_error_color[None]
                )
                black_overlay_pred = renderer.render_mesh(
                    vertices_cr_pred[i].cuda(), black_overlay_pred, colors=pred_error_color[None]
                )

                # write mpjpe on the img
                obs_mpjpe_ = camcoord_metrics["mpjpe"][i]
                text = f"obs mpjpe: {obs_mpjpe_:.1f} ({obs_mpjpe:.1f})"
                cv2.putText(black_overlay_obs, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
                pred_mpjpe_ = self.metric_aggregator["mpjpe"][vid][i]
                text = f"pred mpjpe: {pred_mpjpe_:.1f} ({pred_mpjpe:.1f})"
                if pred_mpjpe_ > obs_mpjpe_:
                    # large error -> purple
                    cv2.putText(black_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2)
                else:
                    # small error -> yellow
                    cv2.putText(black_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 100), 2)
                black = np.concatenate([black_overlay_obs, black_overlay_pred], axis=0)
                ###########################################

                img = np.concatenate([img, black], axis=1)

                writer.append_data(img)
            writer.close()

        if False:  # Visualize incam + global results

            def move_to_start_point(verts):
                "XZ to origin, Start from the ground"
                verts = verts.clone()  # (L, V, 3)
                xz_mean = verts[0].mean(0)[[0, 2]]
                y_min = verts[0, :, [1]].min()
                offset = torch.tensor([[[xz_mean[0], y_min, xz_mean[1]]]]).to(verts)
                verts = verts - offset
                return verts

            verts_incam = pred_c_verts.clone()
            # verts_glob = move_to_start_point(target_ay_verts)  # gt
            verts_glob = move_to_start_point(pred_ay_verts)
            global_R, global_T, global_lights = get_global_cameras_static(verts_glob.cpu())

            # -- rendering code (global version FOV=55) -- #
            vname = batch["meta_render"][0]["name"]
            width, height = batch["meta_render"][0]["width_height"]
            K = batch["meta_render"][0]["K"]
            faces = self.faces_smpl
            out_fn = f"outputs/dump_render_global/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
            writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)

            # two renderers
            renderer_incam = Renderer(width, height, device="cuda", faces=faces, K=K)
            renderer_glob = Renderer(width, height, estimate_focal_length(width, height), device="cuda", faces=faces)

            # imgs
            video_path = batch["meta_render"][0]["video_path"]
            frame_id = batch["meta_render"][0]["frame_id"].cpu().numpy()
            vr = decord.VideoReader(video_path)
            images = vr.get_batch(list(frame_id)).numpy()  # (F, H/4, W/4, 3), uint8, numpy

            # Actual rendering
            cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
            scale = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]].max() * 1.5
            renderer_glob.set_ground(scale, cx.item(), cz.item())
            color = torch.ones(3).float().cuda() * 0.8

            for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
                # incam
                img_overlay_pred = renderer_incam.render_mesh(verts_incam[i].cuda(), images[i], [0.8, 0.8, 0.8])
                if batch["meta_render"][0].get("bbx_xys", None) is not None:  # draw bbox lines
                    bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                    img_overlay_pred = cv2.rectangle(img_overlay_pred, lu_point, rd_point, (255, 178, 102), 2)
                pred_mpjpe_ = self.metric_aggregator["mpjpe"][vid][i]
                text = f"pred mpjpe: {pred_mpjpe_:.1f}"
                cv2.putText(img_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2)

                # glob
                cameras = renderer_glob.create_camera(global_R[i], global_T[i])
                img_glob = renderer_glob.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)

                # write
                img = np.concatenate([img_overlay_pred, img_glob], axis=1)
                writer.append_data(img)
            writer.close()
            print("Done")

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        monitor_metric = "mpjpe"

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        if False:  # debug to make sure the all_gather is correct
            print(f"[RANK {local_rank}/{world_size}]: {self.metric_aggregator[monitor_metric].keys()}")

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # average over all batches
        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        if local_rank == 0:
            Log.info("metrics:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")

        if self.perjoint_metrics:
            metrics_avg = {
                k + "-" + j: np.concatenate(list(v[j].values())).mean()
                for j in self.body_joint_names
                for k, v in self.perjoint_metric_aggregator.items()
            }
            obs_metrics_avg = {
                k + "-" + j: np.concatenate(list(v[j].values())).mean()
                for j in self.body_joint_names
                for k, v in self.perjoint_obs_metric_aggregator.items()
            }
            if local_rank == 0:
                Log.info(
                    "perjoint metrics:\n"
                    + "\n".join(f"{k}: {v:.1f} (obs: {obs_metrics_avg[k]:.1f})" for k, v in metrics_avg.items())
                    + "\n------"
                )

        # print monitored metric per sequence
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 10 if trainer.state.stage == "fit" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    "monitored metric per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric/{k}": v}, step=cur_epoch)

            # Monitor for checkpointing
            pl_module.log(f"monitor/{monitor_metric}", metrics_avg[monitor_metric])

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


def is_target_task(batch):
    task = check_equal_get_one(batch["task"], "task")
    return task == "CAP-Seq"


# Mask out the padded frames
def length_to_mask(lengths, max_len):
    """
    Returns: (B, max_len)
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


cfg_metric_mocap = builds(MetricMocap)
MainStore.store(name="supermotion_mocap", node=cfg_metric_mocap, group=f"callbacks/metric")
