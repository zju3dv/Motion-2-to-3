import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from hmr4d.network.gmd.clip import CLIPLatentEncoder
from hmr4d.network.gmd.mdm_unet import MdmUnet
from hmr4d.network.gmd.gmd_helper import GmdHelper
from hmr4d.network.base_arch.lora import LoRACompatibleLinear
from hmr4d.utils.hml3d import convert_bfj3_to_b263f, convert_hmlvec263_to_motion, convert_motion_to_hmlvec263
from hmr4d.utils.pylogger import Log
from pathlib import Path
import numpy as np
from einops import einsum
from .freeze_unet_utils import FreezeUnetFuncs
from .mr_loraloader import LoraLoader
from einops import rearrange, repeat, einsum
from hmr4d.utils.geo.proj_constraint import constraint
from hmr4d.utils.geo_transform import T_transforms_points, project_p2d


class MrMotionNet(nn.Module, GmdHelper, LoraLoader):
    def __init__(
        self,
        args,
        args_clip,
        args_unet,
        args_freeze,
        args_lora,
    ):
        super().__init__()
        self.args = args
        self.args_freeze = args_freeze
        self.args_lora = args_lora
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Initialize Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt)  # Training is always DDPMScheduler
        self.te_scheduler = self.sch_get_scheduler()

        # ----- Initialize Model ----- #
        self.clip = CLIPLatentEncoder(**args_clip)
        self.clip_proj = LoRACompatibleLinear(self.clip.clip_dim, args_unet.latent_dim)

        # MDM-UNet
        self.unet = MdmUnet(**args_unet)

        # Mean, Std, and projection
        data_dir = Path(__file__).parent.parent / "gmd/data"
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
        if self.args_freeze.clip_proj_trainable is None:
            self.clip_proj.eval()
            self.clip_proj.requires_grad_(False)
        else:
            assert self.args_freeze.clip_proj_trainable == "all"
            self.clip_proj.requires_grad_(True)

    def freeze_unet(self):
        if self.args_freeze.unet_trainable is None:
            self.unet.eval()
            self.unet.requires_grad_(False)
        else:
            FreezeUnetFuncs[self.args_freeze.unet_trainable](self)

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
        hmlvec263 = hmlvec263 * self.hmlvec263_std + self.hmlvec263_mean
        return hmlvec263

    # ========== Training ========== #

    def forward_train(self, inputs):
        outputs = dict()

        hmlvec263 = inputs["gt_hmlvec263"]  # (B, 263, L)
        scheduler = self.tr_scheduler
        B, _, L = hmlvec263.shape  # L is the real length of the trajectory
        drop_clip = self.args.drop_clip  # default = 0.1

        # 1. Encode CLIP embedding
        assert self.args.prompt_latent_type == "saved_embeds"
        prompt_latent = self.clip.encode_image_sequence(saved_embeds=inputs["saved_embeds"])
        if drop_clip > 0 and np.random.random() <= drop_clip:  # dropout
            prompt_latent = torch.zeros_like(prompt_latent)  # behaviour is the same as GMD
        prompt_latent = self.clip_proj(prompt_latent)  # (B, L=1, D)

        # 2. Emphasis-project
        x = self.encode_hmlvec263(hmlvec263)

        # 3. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # predict the noise residual
        model_pred = self.unet(noisy_x, t, encoder_hidden_states=prompt_latent).sample

        prediction_type = scheduler.config.prediction_type
        if prediction_type == "sample":
            target = x
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            assert prediction_type == "epsilon"
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs["loss"] = loss
        return outputs

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
        Similiar to `gmd_traj_net.py`, but with the following differences:
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
        length_ = 224 if self.unet.version == "GMD_traj_release" else length
        x = self.prepare_x(shape=(B, self.unet.input_dim, length_), generator=generator)
        noise = self.randlike_shape(x.shape, generator)

        # 2. Encode CLIP embedding
        if self.args.prompt_latent_type == "text":
            prompt_latent = self.clip.encode_text(inputs["prompt_text"], enable_cfg=enable_cfg)
            prompt_latent = self.clip_proj(prompt_latent)  # (B(*2), L=1, D)
        elif self.args.prompt_latent_type == "image_sequence":
            prompt_latent = self.clip.encode_image_sequence(inputs["prompt_imgs"], enable_cfg=enable_cfg)
            prompt_latent = self.clip_proj(prompt_latent)  # (B(*2), L=1, D)
        elif self.args.prompt_latent_type == "saved_embeds":
            prompt_latent = self.clip.encode_image_sequence(
                saved_embeds=inputs["saved_embeds"],
                enable_cfg=enable_cfg,
            )  # (B(*2), L=1, D)
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

            # *. Inpaint
            if self.args.inpaint.method is not None:
                inpaint_args = (x0_, xprev_, noise, timesteps, i)
                xprev_ = self.compute_inpaint(step_args, inpaint_args, inputs)

            # *. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ori_space = self.decode_hmlvec263(x0_.detach())[..., :length]  # (B, 263, L)
                pred_motion_progress.append(convert_hmlvec263_to_motion(x0_ori_space, abs_3d=True))  # (B, L, 22, 3)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x0_ori_space = self.decode_hmlvec263(x)[..., :length]  # (B, 4, L)
        outputs["pred_motion"] = convert_hmlvec263_to_motion(x0_ori_space, abs_3d=True)  # (B, L, 22, 3)
        outputs["pred_motion_progress"] = torch.stack(pred_motion_progress, dim=1)  # (B, Progress, L, 22, 3)
        return outputs

    # ========== Inpaint ========== #
    def compute_inpaint(self, step_args, inpaint_args, inputs=None):
        method = self.args.inpaint.method
        t_end = self.args.inpaint.t_end
        t_start = self.args.inpaint.t_start

        # Skip if not in the range
        t = step_args[1]
        if not (t_end < t <= t_start):
            x0_, xprev_, noise, timesteps, i = inpaint_args
            return xprev_

        # Do the computation
        opts = self.args.inpaint.get(method, None)
        if method in ["gmd_partial_inpaint"]:
            func = self.inpaint_gmd_partial
        elif method in ["add_noise_hmlvec263"]:
            func = self.inpaint_add_noise_hmlvec263
        elif method in ["inpaint_c1v"]:
            func = self.inpaint_c1v
        return func(step_args, inpaint_args, inputs, opts)

    def inpaint_gmd_partial(self, step_args, inpaint_args, inputs, opts):
        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args
        x0_, xprev_, noise, timesteps, i = inpaint_args

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

    def inpaint_add_noise_hmlvec263(self, step_args, inpaint_args, inputs, opts):
        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args
        x0_, xprev_, noise, timesteps, i = inpaint_args

        # The x0_given is not complete, the inpainting be cannot done with add_noise(x0_given, t_prev)
        length = inputs["cond_hmlvec263"].shape[-1]
        padding = 224 - length if self.unet.version == "GMD_traj_release" else 0
        x0_given = F.pad(inputs["cond_hmlvec263"], (0, padding), value=0.0)  # (B, 263, L)
        x0_given = self.encode_hmlvec263(x0_given)
        x0_given_mask = F.pad(inputs["cond_hmlvec263_mask"], (0, padding), value=0.0)

        assert isinstance(scheduler, DDIMScheduler)
        xprev_given = scheduler.add_noise(x0_given, noise=noise, timesteps=timesteps[i + 1])
        xprev_ = xprev_ * ~x0_given_mask + xprev_given * x0_given_mask

        return xprev_

    def inpaint_c1v(self, step_args, inpaint_args, inputs, opts):
        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args
        x0_, xprev_, noise, timesteps, i = inpaint_args

        pred_hmlvec263 = self.decode_hmlvec263(x0_)  # ( B, 263, L)
        pred_ayf_motion = convert_hmlvec263_to_motion(pred_hmlvec263, abs_3d=True)  # (B, L, 22, 3)
        B, L, J, _ = pred_ayf_motion.shape

        # Get gt c_p2d
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)
        c_p3d = T_transforms_points(T_ayf2c, inputs["gt_motion"], "b c d, b l j d -> b l j c")
        proj_mode = "persp"
        c_p3d_ = rearrange(c_p3d, "b l j c -> b (l j) c")
        K = torch.eye(3, device=c_p3d.device).unsqueeze(0).repeat(B, 1, 1)
        c_p2d_ = project_p2d(c_p3d_, K, is_pinhole=proj_mode == "persp")  # (B, L*22, 2)

        # Do constraint on pred_ayf_motion
        pred_ayf_motion_ = rearrange(pred_ayf_motion, "b l j c -> b (l j) c")
        pred_ayf_motion_ = constraint(pred_ayf_motion_, T_ayf2c, c_p2d_, mode=proj_mode)
        pred_ayf_motion_ = rearrange(pred_ayf_motion_, "b (l j) c -> b l j c", l=L, j=J)

        pred_hmlvec263 = convert_motion_to_hmlvec263(pred_ayf_motion_, return_abs=True)
        x0_inpaint = self.encode_hmlvec263(pred_hmlvec263)

        # Use posterior p(x{t-1} | xt, x0)
        assert scheduler.config.prediction_type == "sample"
        xprev_ = scheduler.step(x0_inpaint, t, x, **extra_step_kwargs).prev_sample

        return xprev_
