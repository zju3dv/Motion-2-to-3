import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from hmr4d.network.gmd.clip import CLIPLatentEncoder
from hmr4d.network.gmd.mdm_unet import MdmUnet
from hmr4d.utils.hml3d import recover_root_rot_pos, convert_bf3_to_b4f
from .gmd_helper import GmdHelper


class GmdTrajNet(nn.Module, GmdHelper):
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
        self.inpaint = args.inpaint
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Initialize Scheduler ----- #
        # self.tr_scheduler = DDPMScheduler(**args.ddpmscheduler_cfg)
        self.te_scheduler = self.sch_get_scheduler()

        # ----- Initialize Model ----- #
        self.clip = CLIPLatentEncoder(**args_clip)
        self.clip_proj = nn.Linear(self.clip.clip_dim, args_unet.latent_dim)

        # MDM-UNet
        self.unet = MdmUnet(**args_unet)

        # fmt: off
        self.register_buffer("hmlvec4_std", torch.tensor([0.64632845, 0.72565746, 0.72565746, 0.15380478]).reshape(4, 1), False)
        self.register_buffer("hmlvec4_mean", torch.tensor([-1.0136745e-03, 3.9973247e-04, 3.1452757e-01, 9.3865341e-01]).reshape(4, 1), False)
        # fmt: on

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

    def encode_hmlvec4(self, hmlvec4):
        assert hmlvec4.shape[-2] == 4, "Shape should be (B, 4, L)"
        return (hmlvec4 - self.hmlvec4_mean) / self.hmlvec4_std

    def decode_hmlvec4(self, hmlvec4):
        assert hmlvec4.shape[-2] == 4, "Shape should be (B, 4, L)"
        return hmlvec4 * self.hmlvec4_std + self.hmlvec4_mean

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
        Generate Root-Traj (B, L, 3) from the given conditions (Text, XZ, etc.)
        Note that the diffusion is done in the HumanML-Vec-4 space.
        The representation is in the coordinate xz-y, which aligns to the SMPL coordinate.
        L should be less than 224, since the UNet is trained on 224.
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
        noise = self.randlike_shape(x.shape, generator)

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
        pred_traj_progress = []  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            step_args = [x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg]
            _, x0_, xprev_ = self.denoising_step(*step_args)

            # 2. Classifier-guidance
            if self.args.classifier_guidance.method is not None:
                xprev_ += self.compute_classifier_guidance(step_args, inputs=inputs)

            # 3. Inpaint
            if i < len(timesteps) - 1 and self.inpaint.t_end <= t <= self.inpaint.t_start:
                x0_given = self.encode_hmlvec4(convert_bf3_to_b4f(inputs["cond_traj"], pad_f_to=224))
                x0_given_mask = convert_bf3_to_b4f(inputs["cond_traj_mask"], pad_f_to=224)
                xprev_given = scheduler.add_noise(x0_given, noise=noise, timesteps=timesteps[i + 1])
                xprev_ = xprev_ * ~x0_given_mask + xprev_given * x0_given_mask

            # 4. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ = self.decode_hmlvec4(x0_)[..., :length]  # (B, 4, L)
                _, pred_traj = recover_root_rot_pos(x0_.permute(0, 2, 1), abs_3d=True)  # (B, L, 3)
                pred_traj_progress.append(pred_traj)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        pred_hmlvec4 = self.decode_hmlvec4(x)[..., :length]  # (B, 4, L)
        outputs["pred_hmlvec4"] = pred_hmlvec4  # (B, 4, L)
        _, pred_traj = recover_root_rot_pos(pred_hmlvec4.permute(0, 2, 1), abs_3d=True)  # (B, L, 3)
        outputs["pred_traj"] = pred_traj  # (B, L, 3)
        outputs["pred_traj_progress"] = torch.stack(pred_traj_progress, dim=1)
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
        if method in ["cg_gmd_x0_v1", "cg_gmd_x0_v2"]:
            opts = self.args.classifier_guidance.get(method)
            return self.cg_gmd_x0(step_args, inputs, opts)
        else:
            raise NotImplementedError

    @torch.inference_mode(mode=False)
    def cg_gmd_x0(self, step_args, inputs, opts):
        """
        Use the prediction of unet to guide the diffusion.
        """
        x, t, prompt_latent, scheduler, extra_step_kwargs, enable_cfg = step_args
        # Setup the target
        if opts.traj_as_rkpt:
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

        if opts.get("reweight_x0", False):
            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            fac = torch.sqrt(beta_prod_t)
            x0_ = x0_ * (fac) + x * (1 - fac)

        # compute the loss
        x0_ = self.decode_hmlvec4(x0_)  # (B, 4, L)
        _, pred_traj = recover_root_rot_pos(x0_.permute(0, 2, 1), abs_3d=True)  # (B, L, 3)
        loss_sum = ((pred_traj - cond_rkpt) ** 2 * cond_rkpt_mask.float()).sum()
        loss = loss_sum / rkpt_sum  # each target is normalized by the number of valid points
        grads = torch.autograd.grad(-loss, x)[0]

        # compute update values
        cg_update = self.sch_get_var(t) * opts.scale * grads
        return cg_update.detach()
