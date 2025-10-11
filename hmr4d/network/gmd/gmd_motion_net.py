import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from hmr4d.network.gmd.clip import CLIPLatentEncoder
from hmr4d.network.gmd.mdm_unet import MdmUnet
from .gmd_helper import GmdHelper
from hmr4d.utils.hml3d import convert_bfj3_to_b263f, recover_from_ric, recover_root_rot_pos, convert_bf3_to_b4f

from pathlib import Path
import numpy as np
from einops import einsum


class GmdMotionNet(nn.Module, GmdHelper):
    def __init__(
        self,
        args,
        args_clip,
        args_unet,
        args_freeze,
    ):
        super().__init__()
        self.args = args
        self.args_freeze = args_freeze
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Initialize Scheduler ----- #
        # self.tr_scheduler = DDPMScheduler(**args.ddpmscheduler_cfg)
        self.te_scheduler = self.sch_get_scheduler()

        # ----- Initialize Model ----- #
        self.clip = CLIPLatentEncoder(**args_clip)
        self.clip_proj = nn.Linear(self.clip.clip_dim, args_unet.latent_dim)

        # MDM-UNet
        self.unet = MdmUnet(**args_unet)

        # Mean, Std, and projection
        data_dir = Path(__file__).parent / "data"
        hmlvec263_mean = np.load(data_dir / "Mean_abs_3d.npy", allow_pickle=True)
        hmlvec263_std = np.load(data_dir / "Std_abs_3d.npy", allow_pickle=True)
        self.register_buffer("hmlvec263_mean", torch.from_numpy(hmlvec263_mean).reshape(263, 1), False)
        self.register_buffer("hmlvec263_std", torch.from_numpy(hmlvec263_std).reshape(263, 1), False)

        # random projection + scaling
        emph_proj = torch.from_numpy(np.load(data_dir / "rand_proj.npy", allow_pickle=True))  # (263, 263)
        self.register_buffer("emph_proj", emph_proj, False)
        self.register_buffer("inv_emph_proj", emph_proj.inverse(), False)

        # ----- Freeze ----- #
        self.freeze_clip()
        self.freeze_clip_proj()
        self.freeze_unet()

    def freeze_clip(self):
        assert self.args_freeze.clip_trainable is None
        self.clip.eval()
        self.clip.requires_grad_(False)

    def freeze_clip_proj(self):
        assert self.args_freeze.clip_proj_trainable is None
        self.clip_proj.eval()
        self.clip_proj.requires_grad_(False)

    def freeze_unet(self):
        assert self.args_freeze.unet_trainable is None
        self.unet.eval()
        self.unet.requires_grad_(False)

    def encode_hmlvec263(self, hmlvec263):
        """
        HmlVec263 is in the absolute 3D space.
        This function first normalizes it, then uses emphasis-project (proposed by GMD).
        """
        assert hmlvec263.shape[-2] == 263, "Shape should be (B, 263, L)"
        hmlvec263 = (hmlvec263 - self.hmlvec263_mean) / self.hmlvec263_std
        hmlvec263_proj = einsum(hmlvec263, self.emph_proj, "b d l , d c -> b c l")
        return hmlvec263_proj

    def decode_hmlvec263(self, hmlvec263_proj):
        """Reverse process of encode_hmlvec263"""
        assert hmlvec263_proj.shape[-2] == 263, "Shape should be (B, 263, L)"
        hmlvec263 = einsum(hmlvec263_proj, self.inv_emph_proj, "b d l , d c -> b c l")
        hmlvec263 = hmlvec263 * self.hmlvec263_std + +self.hmlvec263_mean
        return hmlvec263

    # ========== Sample ========== #
    def denoising_step(self, x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg):
        # expand the x if we are doing classifier free guidance
        x_model_input = torch.cat([x] * 2) if enable_cfg else x
        x_model_input = scheduler.scale_model_input(x_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(x_model_input, t, encoder_hidden_states=prompt_latent).sample

        # classifier-free guidance
        if enable_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_{t-1} and the original sample x_0
        scheduler_out = scheduler.step(noise_pred, t, x, **extra_step_kwargs)
        x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample
        return noise_pred, x0_, xprev_

    @torch.inference_mode()
    def forward_sample(self, inputs):
        """
        Similiar to gmd_traj_net.py, but with the following differences:
        Generate Motion (22 joints) (B, L, 22, 3)
        The diffusion is done in the HumanML-Vec-263 space.
        """
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = self.guidance_scale >= 1.0
        scheduler = self.te_scheduler

        B = inputs["length"].shape[0]
        length = inputs["length"][0]  # scalar
        assert all(inputs["length"] == length), "The length of the trajectory should be the same."
        assert length < 224, "The length of the trajectory should be less than 224."

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.unet.input_dim, 224), generator=generator)

        # 2. Encode CLIP embedding
        prompt_latent = self.clip.encode_text(inputs["prompt_text"], enable_cfg=enable_cfg)
        prompt_latent = self.clip_proj(prompt_latent)  # (B(*2), L=1, D)

        # 3. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)  # for scheduler.step()
        pred_motion_progress = []  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            step_args = [x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg]
            _, x0_, xprev_ = self.denoising_step(*step_args)

            # 2. Classifier-guidance
            if self.args.classifier_guidance.method is not None:
                xprev_ += self.compute_classifier_guidance(step_args, inputs=inputs)

            # 3. Inpaint
            if self.args.inpaint.method is not None:
                xprev_ = self.compute_inpaint(step_args, x0_, xprev_, inputs)

            # 4. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ori_space = self.decode_hmlvec263(x0_.detach())[..., :length]  # (B, 263, L)
                pred_motion = recover_from_ric(x0_ori_space.permute(0, 2, 1), abs_3d=True)
                pred_motion_progress.append(pred_motion)  # (B, L, 22, 3)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x0_ori_space = self.decode_hmlvec263(x)[..., :length]  # (B, 4, L)
        pred_motion = recover_from_ric(x0_ori_space.permute(0, 2, 1), abs_3d=True)
        outputs["pred_motion"] = pred_motion  # (B, L, 22, 3)
        outputs["pred_motion_progress"] = torch.stack(pred_motion_progress, dim=1)  # (B, Progress, L, 22, 3)
        return outputs

    # ========== Classifier-Guidance ========== #
    @torch.inference_mode(mode=False)
    def compute_classifier_guidance(self, step_args, inputs=None):
        method = self.args.classifier_guidance.method
        t_end = self.args.classifier_guidance.t_end
        t_start = self.args.classifier_guidance.t_start

        # Skip if not in the range
        t = step_args[1]
        if not (t_end < t <= t_start):
            return 0

        # Do the computation
        if method in ["cg_gmd_x0"]:
            opts = self.args.classifier_guidance.get(method)
            return self.cg_gmd_x0(step_args, inputs, opts)
        else:
            raise NotImplementedError

    def cg_gmd_x0(self, step_args, inputs, opts):
        """
        Use the prediction of unet to guide the diffusion.
        """
        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args
        # Setup the target
        if self.xz_guidance.traj_as_rkpt:
            length = inputs["cond_traj"].shape[-2]
            cond_rkpt = F.pad(inputs["cond_traj"], (0, 0, 0, 224 - length))
            cond_rkpt_mask = F.pad(inputs["cond_traj_mask"], (0, 0, 0, 224 - length))
        else:
            length = inputs["cond_rkpt"].shape[-2]
            cond_rkpt = F.pad(inputs["cond_rkpt"], (0, 0, 0, 224 - length))
            cond_rkpt_mask = F.pad(inputs["cond_rkpt_mask"], (0, 0, 0, 224 - length))
        rkpt_sum = cond_rkpt_mask.float().sum()
        if rkpt_sum == 0:
            return 0

        # Clone the input and make it requires_grad
        x = x.clone().requires_grad_()
        noise_pred, x0_, xprev_ = self.denoising_step(x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg)

        # compute the loss
        x0_ori_space = self.decode_hmlvec263(x0_)  # (B, 263, L)
        _, pred_traj = recover_root_rot_pos(x0_ori_space.permute(0, 2, 1), abs_3d=True)  # (B, L, 263)
        loss_sum = ((pred_traj - cond_rkpt) ** 2 * cond_rkpt_mask.float()).sum()
        loss = loss_sum / rkpt_sum  # each target is normalized by the number of valid points
        grads = torch.autograd.grad(-loss, x)[0]

        # compute update values
        cg_update = self.sch_get_var(t) * opts.scale * grads
        return cg_update.detach()

    # ========== Inpaint ========== #
    def compute_inpaint(self, step_args, x0_, xprev_, inputs):
        method = self.args.inpaint.method
        assert method == "gmd_partial_inpaint"
        t_end = self.args.inpaint.t_end
        t_start = self.args.inpaint.t_start

        # Skip if not in the range
        t = step_args[1]
        if not (t_end < t <= t_start):
            return xprev_

        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args

        # The x0_given is not complete, the inpainting be cannot done with add_noise(x0_given, t_prev)
        x0_given = convert_bfj3_to_b263f(inputs["cond_motion"], pad_f_to=224)  # (B, 263, L)
        x0_given_mask = convert_bfj3_to_b263f(inputs["cond_motion_mask"], pad_f_to=224)  # (B, 263, L)
        # x0_given = F.pad(inputs["cond_hmlvec4"], (0, 224 - 120, 0, 263 - 4))
        # x0_given_mask = F.pad(inputs["cond_hmlvec4_mask"], (0, 224 - 120, 0, 263 - 4))

        # GMD: inpaint the predicted x0 (this seems not to be a common practice in DMs)
        # NOTE: I think the x_t is also noisy, so the inpainting may be biased.
        x0_ori_space = self.decode_hmlvec263(x0_)  # (B, 263, L)
        x0_ori_space = x0_ori_space * ~x0_given_mask + x0_given * x0_given_mask
        x0_inpaint = self.encode_hmlvec263(x0_ori_space)

        # Use posterior p(x{t-1} | xt, x0)
        if isinstance(scheduler, DDPMScheduler):
            prev_t = scheduler.previous_timestep(t)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
            current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
            xprev_mean_from_x0_inpaint = pred_original_sample_coeff * x0_inpaint + current_sample_coeff * x
        else:
            assert isinstance(scheduler, DDIMScheduler)
            xprev_mean_from_x0_inpaint = scheduler.step(
                x0_inpaint, t, x, variance_noise=0, **extra_step_kwargs
            ).prev_sample

        xprev_mean_from_x0_inpaint_ori_space = self.decode_hmlvec263(xprev_mean_from_x0_inpaint)
        xprev_ori_space = self.decode_hmlvec263(xprev_)
        xprev_ori_space = xprev_ori_space * ~x0_given_mask + xprev_mean_from_x0_inpaint_ori_space * x0_given_mask
        xprev_ = self.encode_hmlvec263(xprev_ori_space)

        return xprev_
