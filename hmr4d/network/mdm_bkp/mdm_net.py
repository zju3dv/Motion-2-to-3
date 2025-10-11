import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from hmr4d.network.gmd.clip import CLIPLatentEncoder

from hmr4d.network.gmd.mdm_unet import MdmUnet
from hmr4d.network.mdm.mdm import MDM
from hmr4d.network.gmd.gmd_helper import GmdHelper

from pathlib import Path
import numpy as np
from einops import einsum, rearrange, repeat

import hmr4d.network.mdm.statistics as statistics


class MDMNet(nn.Module, GmdHelper):
    def __init__(self, args, args_unet, args_clip):
        super().__init__()
        self.args = args
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Initialize Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt)  # Training is always DDPMScheduler
        self.te_scheduler = DDPMScheduler(**args.scheduler_opt)  # 2D Motion sampler

        # ----- Initialize Model ----- #
        self.unet = eval(args_unet.model)(**args_unet)

        # Mean, Std
        stats = getattr(statistics, self.args.stats_name)
        self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(stats["std"]).float(), False)

        self.clip = CLIPLatentEncoder(**args_clip)

        # ----- Freeze ----- #
        self.freeze_clip()

    def freeze_clip(self):
        self.clip.eval()
        self.clip.requires_grad_(False)

    def encode_motion(self, motion):
        """
        this funciton normalize this into standard gaussian distribution
        """
        mean_shape = self.mean.shape  # (263)
        assert motion.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        x = (motion - self.mean) / self.std
        return x

    def decode_motion(self, x):
        """Reverse process of encode_motion"""
        mean_shape = self.mean.shape  # (263)
        assert x.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        motion = x * self.std + self.mean
        return motion

    # ========== Training ========== #

    def forward_train(self, inputs):
        outputs = dict()
        motion = inputs["hmlvec"]  # (B, L, 263)
        scheduler = self.tr_scheduler
        B, L, _ = motion.shape  # L is the real length of the trajectory

        # *. Encoding
        x = self.encode_motion(motion)
        x = rearrange(x, "b l c -> b c l")  # input of unet

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Encode CLIP embedding
        # FIXME Find that too long text will print sth here
        prompt_latent = self.clip.encode_text(inputs["text"], enable_cfg=False)

        # predict the noise residual
        model_output = self.unet(noisy_x, t, prompt_latent=prompt_latent, **inputs)
        model_pred = model_output.sample
        mask = model_output.mask

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

        outputs["loss"] = loss
        return outputs

    # ========== Sample ========== #
    def denoising_step(self, x, t, model_kwargs, scheduler, extra_step_kwargs, enable_cfg):
        # expand the x if we are doing classifier free guidance
        x_model_input = torch.cat([x] * 2) if enable_cfg else x
        x_model_input = scheduler.scale_model_input(x_model_input, t)

        text = model_kwargs["text"]

        prompt_latent = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B, 1, D)

        # predict the noise residual
        noise_pred = self.unet(x_model_input, t, prompt_latent=prompt_latent, **model_kwargs).sample

        # classifier-free guidance
        if enable_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_{t-1} and the original sample x_0
        scheduler_out = scheduler.step(noise_pred, t, x, **extra_step_kwargs)
        x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample
        return noise_pred, x0_, xprev_

    def get_model_kwargs(self, inputs):
        model_kwargs = {
            "text": inputs["text"],
            "length": inputs["length"],
            "enable_cfg": True,
        }
        return model_kwargs

    @torch.inference_mode()
    def forward_sample(self, inputs):
        """Generate Motion (22 joints) (B, L, 263)"""
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        scheduler = self.te_scheduler

        B = inputs["length"].shape[0]
        L = inputs["length"][0]  # scalar
        # TODO: Does not support variable length
        assert all(inputs["length"] == L), "The length of the trajectory should be the same."
        assert L < 224, "The length of the trajectory should be less than 224."

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.unet.input_dim, L), generator=generator)
        noise = self.randlike_shape(x.shape, generator)

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)  # for scheduler.step()
        pred_progress = []  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.get_model_kwargs(inputs)
            enable_cfg = model_kwargs["enable_cfg"]
            step_args = [x, t, model_kwargs, scheduler, extra_step_kwargs, enable_cfg]
            _, x0_, xprev_ = self.denoising_step(*step_args)

            # *. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ = rearrange(x0_, "b c l -> b l c")
                x0_ori_space = self.decode_motion(x0_)  # (B, L, 263)
                pred_progress.append(x0_ori_space)  # (B, L, 263)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x = rearrange(x, "b c l -> b l c")
        x0_ori_space = self.decode_motion(x)  # (B, L, 263)
        outputs["pred_motion"] = x0_ori_space  # (B, L, 263)
        outputs["pred_motion_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, 263)
        return outputs
