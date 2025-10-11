import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase, SMPLEnDecoder

from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.geo.triangulation import triangulate_2d_3d
from hmr4d.utils.geo.optim import find_wst_transform, update_pred_with_wst
from hmr4d.utils.geo_transform import apply_T_on_points, transform_mat
from hmr4d.utils.geo_transform import kabsch_algorithm_batch, similarity_transform_batch
from hmr4d.utils.filter import smooth_pose_oneeurofilter, smooth_pose_savgol, smooth_pose_gaussian
from pytorch3d.ops import efficient_pnp
from hmr4d.utils.geo_transform import ransac_PnP_batch, transform_mat
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from scipy.ndimage._filters import _gaussian_kernel1d


# We use HMR2 feature instead of clip image feature
class MotionSMPLPriorPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser3d, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser3d: denoiser3d network
        """
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        num_visualize = min(args.num_visualize, self.num_inference_steps)
        self.record_interval = self.num_inference_steps // num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = instantiate(args.scheduler_opt_sample)

        # ----- Networks ----- #
        self.clip = instantiate(args_clip, _recursive_=False)
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        Log.info(self.denoiser3d)

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: SMPLEnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode
        self.decoder_motion3d = self.data_endecoder.decode
        self.decoder_smpl_to_joints = self.data_endecoder.decode_joints
        # self.decoder_smpl_to_joints = self.data_endecoder.decode_joints_from_vel
        self.decoder_root_to_joints = self.data_endecoder.decode_joints_from_root
        # self.decoder_root_to_joints = self.data_endecoder.decode_joints_from_root_vel

        # ----- Freeze ----- #
        self.freeze_clip()

    def freeze_clip(self):
        self.clip.eval()
        self.clip.requires_grad_(False)

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs(x, timesteps, clip_text, f_imgseq, inputs, enable_cfg, **kwargs):
        """override this if you want to add more kwargs"""
        # supermotion/decoder_multiple_crossattn
        length = torch.cat([inputs["length"]] * 2) if enable_cfg else inputs["length"]
        f_imgseq_zero = torch.zeros_like(f_imgseq)
        f_imgseq = torch.cat([f_imgseq_zero, f_imgseq]) if enable_cfg else f_imgseq
        model_kwargs = dict(
            x=x,
            timesteps=timesteps,
            length=length,
            f_text=clip_text.f_text,
            f_text_length=clip_text.f_text_length,
            f_imgseq=f_imgseq,
        )
        return model_kwargs

    def forward_train(self, inputs):
        outputs = dict()
        motion = inputs["smplpose"]  # (B, L, 69)
        length = inputs["length"]  # (B,) effective length of each sample
        scheduler = self.tr_scheduler
        B, L, _ = motion.shape

        # *. Encoding
        x = self.encoder_motion3d(motion, length=length)  # (B, C, L)

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Encode CLIP embedding
        assert self.training
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]  # (B, L, C=1024)
        text, f_imgseq = randomly_set_null_condition(text, f_imgseq)
        clip_text = self.clip.encode_text(text, enable_cfg=False)  # (B, 77, D)

        # allow custom kwargs
        model_kwargs = self.build_model_kwargs(
            x=noisy_x,
            timesteps=t,
            clip_text=clip_text,
            f_imgseq=f_imgseq,
            inputs=inputs,
            enable_cfg=False,
        )
        model_output = self.denoiser3d(**model_kwargs)
        model_pred = model_output.sample
        mask = model_output.mask  # (B, 1, L)

        total_loss = 0
        ########### Simple objective loss #############
        prediction_type = scheduler.config.prediction_type
        if prediction_type == "sample":
            target = x
        else:
            assert prediction_type == "epsilon"
            target = noise
        if mask is not None:
            model_pred = model_pred * mask
            target = target * mask
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        total_loss += loss
        outputs["simple_loss"] = loss
        #####################################

        pos_weight = self.args.loss_pos_weight
        vel_weight = self.args.loss_vel_weight
        contact_weight = self.args.loss_vel_weight
        fk_weight = self.args.loss_fk_weight
        selfconsistency_weight = self.args.loss_selfconsistency_weight
        crossconsistency_weight = self.args.loss_crossconsistency_weight

        gt_pos = self.data_endecoder.get_ayfz_joints(motion)  # (B, L, J, 3)
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]  # (B, L-1, J, 3)
        contact_label = self.data_endecoder.detect_foot_contact(gt_pos)[:, :-1]  # (B, L-1, 4)
        contact_label = contact_label[..., None]  # (B, L-1, 4, 1)

        pos_mask = mask[:, 0, :, None, None]  # (B, L, 1, 1)
        vel_mask = pos_mask[:, 1:]  # (B, L-1, 1, 1)
        contact_mask = vel_mask  # (B, L-1, 1, 1)

        gt_pos = gt_pos * pos_mask
        gt_vel = gt_vel * vel_mask
        contact_label = contact_label * contact_mask

        ########### Forward kinematics loss #############
        fk_dict = {
            "smpl": self.data_endecoder.decode_joints,
            "smplvel": self.data_endecoder.decode_joints_from_vel,
            "root": self.data_endecoder.decode_joints_from_root,
            "rootvel": self.data_endecoder.decode_joints_from_root_vel,
        }
        fk_out_dict = {}
        fk_loss_dict = {}

        N_fk = len(fk_dict)
        for k, v in fk_dict.items():
            pred_fk_pos = v(model_pred)  # (B, L, J, 3)
            pred_fk_pos = pred_fk_pos * pos_mask
            pred_fk_vel = pred_fk_pos[:, 1:] - pred_fk_pos[:, :-1]  # (B, L-1, J, 3)
            pred_fk_vel = pred_fk_vel * vel_mask
            pred_fkfoot_vel = self.data_endecoder.get_foot_vel(pred_fk_vel)  # (B, L-1, 4, 3)
            pred_fkfoot_vel_square = pred_fkfoot_vel * pred_fkfoot_vel * contact_mask

            fkpos_loss = F.mse_loss(pred_fk_pos.float(), gt_pos.float(), reduction="mean")
            fkvel_loss = F.mse_loss(pred_fk_vel.float(), gt_vel.float(), reduction="mean")
            fkcontact_loss = (contact_label * pred_fkfoot_vel_square).mean()

            fk_loss_dict[k + "_fkpos_loss"] = fkpos_loss
            fk_loss_dict[k + "_fkvel_loss"] = fkvel_loss
            fk_loss_dict[k + "_fkcontact_loss"] = fkcontact_loss
            fk_out_dict[k + "_fkpos"] = pred_fk_pos
            fk_out_dict[k + "_fkvel"] = pred_fk_vel
            fk_weight_ = getattr(fk_weight, k)

            total_loss += fk_weight_ * pos_weight * fkpos_loss / N_fk
            total_loss += fk_weight_ * vel_weight * fkvel_loss / N_fk
            total_loss += fk_weight_ * contact_weight * fkcontact_loss / N_fk
        #####################################

        #########3 Absolute and relative aligned forward kinematics ########
        selfconsistency_group = ["smpl", "root"]
        N_selfconsistency = len(selfconsistency_group)
        for k in selfconsistency_group:
            fkpos_loss = F.mse_loss(
                fk_out_dict[f"{k}_fkpos"].float(), fk_out_dict[f"{k}vel_fkpos"].float(), reduction="mean"
            )
            fkvel_loss = F.mse_loss(
                fk_out_dict[f"{k}_fkvel"].float(), fk_out_dict[f"{k}vel_fkvel"].float(), reduction="mean"
            )

            fk_loss_dict[f"selfconsistency_{k}_fkpos_loss"] = fkpos_loss
            fk_loss_dict[f"selfconsistency_{k}_fkvel_loss"] = fkvel_loss
            total_loss += selfconsistency_weight * pos_weight * fkpos_loss / N_selfconsistency
            total_loss += selfconsistency_weight * vel_weight * fkvel_loss / N_selfconsistency
        #####################################

        ######### SMPL and skeleton realtive aligned forward kinematics ##########
        crossconsistency_group = [["smplvel", "root"]]
        N_crossconsistency = len(crossconsistency_group)
        for k1, k2 in crossconsistency_group:
            fkpos_loss = F.mse_loss(
                fk_out_dict[f"{k1}_fkpos"].float(), fk_out_dict[f"{k2}_fkpos"].float(), reduction="mean"
            )
            fkvel_loss = F.mse_loss(
                fk_out_dict[f"{k1}_fkvel"].float(), fk_out_dict[f"{k2}_fkvel"].float(), reduction="mean"
            )

            fk_loss_dict[f"crossconsistency_{k1}_{k2}_fkpos_loss"] = fkpos_loss
            fk_loss_dict[f"crossconsistency_{k1}_{k2}_fkvel_loss"] = fkvel_loss
            total_loss += crossconsistency_weight * pos_weight * fkpos_loss / N_crossconsistency
            total_loss += crossconsistency_weight * vel_weight * fkvel_loss / N_crossconsistency
        #####################################

        outputs["loss"] = total_loss
        outputs.update(fk_loss_dict)
        return outputs

    def cfg_denoise_func(self, denoiser, model_kwargs, scheduler, enable_cfg):
        x = model_kwargs.pop("x")
        t = model_kwargs.pop("timesteps")

        # expand the x if we are doing classifier free guidance
        x_model_input = torch.cat([x] * 2) if enable_cfg else x
        x_model_input = scheduler.scale_model_input(x_model_input, t)

        # predict
        denoiser_out = denoiser(x_model_input, t, **model_kwargs)
        noise_pred = denoiser_out.sample

        # classifier-free guidance
        if enable_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # a special case since our motion prior is predicting x0 # TODO: extra check may be needed
        x0_ = noise_pred
        return x0_

    # ========== Sample ========== #
    def forward_sample(self, inputs):
        """Generate Motion (22 joints) (B, L, 263)"""
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.te_scheduler

        length = inputs["length"]
        B = inputs["length"].shape[0]

        if "mm_num_repeats" in self.args:
            mm_num_repeats = self.args.mm_num_repeats
            assert B == 1, "Multimodality usually uses batchsize=1"
            for k in inputs.keys():
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = torch.cat([inputs[k]] * mm_num_repeats, dim=0)
                elif isinstance(inputs[k], list):
                    inputs[k] = inputs[k] * mm_num_repeats
                elif k == "B":
                    inputs[k] = inputs[k] * mm_num_repeats
                    B = B * mm_num_repeats
                elif k == "generator":
                    pass
                else:
                    pass

        max_L = inputs["smplpose"].shape[1]  # B, L, c

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.denoiser3d.input_dim, max_L), generator=generator)
        gt_x0 = self.encoder_motion3d(inputs["smplpose"], length=length)  # (B, C, L)

        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]

        if not enable_cfg:
            text = ["" for _ in range(len(text))]
            f_imgseq = torch.zeros_like(f_imgseq)

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B, 77, D)

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator)  # for scheduler.step()
        pred_progress = []  # for visualization
        pred_joints_progress = []  # for visualization
        pred_joints_from_vel_progress = []  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                clip_text=clip_text,
                f_imgseq=f_imgseq,
                inputs=inputs,
                enable_cfg=enable_cfg,
            )
            x0_ = self.cfg_denoise_func(self.denoiser3d, model_kwargs, scheduler, enable_cfg)
            # Set padding value zero NOTE: not sure about this
            x0_ = self.data_endecoder.set_default_padding(x0_, length)

            # x0_[:, 25:] = gt_x0[:, 25:]  # assign local features

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                # x0_ = gt_x0  # NOTE: for debug
                x0_ori_space = self.decoder_motion3d(x0_)  # (B, L, c)
                pred_progress.append(x0_ori_space)  # (B, L, c)
                x0_joints = self.decoder_smpl_to_joints(x0_)  # (B, L, J, 3)
                pred_joints_progress.append(x0_joints)  # (B, L, J, 3)
                x0_joints_from_vel = self.decoder_root_to_joints(x0_)  # (B, L, J, 3)
                pred_joints_from_vel_progress.append(x0_joints_from_vel)  # (B, L, J, 3)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x0_ori_space = self.decoder_motion3d(x)  # (B, L, c)
        x0_joints = self.decoder_smpl_to_joints(x)  # (B, L, J, 3)
        x0_joints_from_vel = self.decoder_root_to_joints(x)  # (B, L, J, 3)
        outputs["pred_smplpose"] = x0_ori_space  # (B, L, c)
        outputs["pred_smplpose_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, c)
        outputs["pred_ayfz_motion"] = x0_joints  # (B, L, J, 3)
        outputs["pred_ayfz_motion_progress"] = torch.stack(pred_joints_progress, dim=1)  # (B, Progress, L, J, 3)
        outputs["pred_ayfz_motion_from_vel"] = x0_joints_from_vel  # (B, L, J, 3)
        outputs["pred_ayfz_motion_from_vel_progress"] = torch.stack(pred_joints_from_vel_progress, dim=1)
        gt_x0_joints = self.data_endecoder.get_ayfz_joints(inputs["smplpose"])
        outputs["gt_ayfz_motion"] = gt_x0_joints
        return outputs


def randomly_set_null_condition(text, f_imgseq):
    """
    Args:
        text: List of str
        f_imgseq: (B, L, C)
    """
    # To support classifier-free guidance, randomly set-to-unconditioned
    B = len(text)
    uncond_prob = 0.1

    # text
    text_mask = torch.rand(B) < uncond_prob
    text_ = ["" if m else t for m, t in zip(text_mask, text)]

    # drop image feature or drop some of the sequences
    f_img_mask = torch.rand(B) < uncond_prob
    f_imgseq_mask = torch.rand(B, f_imgseq.shape[1]) < uncond_prob
    f_imgseq = f_imgseq.clone()
    f_imgseq[f_img_mask] = 0.0
    f_imgseq[f_imgseq_mask] = 0.0

    return text_, f_imgseq
