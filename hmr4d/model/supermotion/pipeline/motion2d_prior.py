import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase

from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.geo.triangulation import triangulate_2d_3d


# We use HMR2 feature instead of clip image feature
class Motion2DPriorPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser2d, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser2d: denoiser2d network
        """
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = instantiate(args.scheduler_opt_sample)

        # ----- Networks ----- #
        self.clip = instantiate(args_clip, _recursive_=False)
        self.denoiser2d = instantiate(args_denoiser2d, _recursive_=False)
        Log.info(self.denoiser2d)

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder.encode
        self.decoder_motion2d = self.data_endecoder.decode

        # ----- Freeze ----- #
        self.freeze_clip()

    def freeze_clip(self):
        self.clip.eval()
        self.clip.requires_grad_(False)

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs(x, timesteps, clip_text, f_imgseq, f_camext, inputs, enable_cfg, **kwargs):
        """override this if you want to add more kwargs"""
        # supermotion/decoder_multiple_crossattn
        length = torch.cat([inputs["length"]] * 2) if enable_cfg else inputs["length"]
        f_imgseq_zero = torch.zeros_like(f_imgseq)
        f_imgseq = torch.cat([f_imgseq_zero, f_imgseq]) if enable_cfg else f_imgseq
        f_camext_zero = torch.zeros_like(f_camext)
        f_camext = torch.cat([f_camext_zero, f_camext]) if enable_cfg else f_camext
        model_kwargs = dict(
            x=x,
            timesteps=timesteps,
            length=length,
            f_text=clip_text.f_text,
            f_text_length=clip_text.f_text_length,
            f_imgseq=f_imgseq,
            f_camext=f_camext,
        )
        return model_kwargs

    def forward_train(self, inputs):
        outputs = dict()
        motion = inputs["gt_motion2d"]  # (B, L, J, 3)
        scheduler = self.tr_scheduler
        B, L, J, _ = motion.shape

        # *. Encoding
        x = self.encoder_motion2d(motion)  # (B, C, L)

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Encode CLIP embedding
        assert self.training
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]  # (B, L, C=1024)
        f_camext = inputs["cam_ext"]  # (B, 1, 4)
        text, f_imgseq, f_camext = randomly_set_null_condition(text, f_imgseq, f_camext)
        clip_text = self.clip.encode_text(text, enable_cfg=False)  # (B, 77, D)

        # allow custom kwargs
        model_kwargs = self.build_model_kwargs(
            x=noisy_x,
            timesteps=t,
            clip_text=clip_text,
            f_imgseq=f_imgseq,
            f_camext=f_camext,
            inputs=inputs,
            enable_cfg=False,
        )
        model_output = self.denoiser2d(**model_kwargs)
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

    # ========== Sample 2D ========== #
    def forward_sample(self, inputs):
        """Generate Motion (22 joints) (B, L, 22 * 2)"""
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.te_scheduler

        B = inputs["length"].shape[0]
        max_L = inputs["gt_motion2d"].shape[1]  # B, L, J, 2

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.denoiser2d.input_dim, max_L), generator=generator)

        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]
        f_camext = inputs["cam_ext"]  # (B, 1, 4)

        if not enable_cfg:
            text = ["" for _ in range(len(text))]
            f_imgseq = torch.zeros_like(f_imgseq)
            f_camext = torch.zeros_like(f_camext)

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B, 77, D)

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator)  # for scheduler.step()
        pred_progress = []  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                clip_text=clip_text,
                f_imgseq=f_imgseq,
                f_camext=f_camext,
                inputs=inputs,
                enable_cfg=enable_cfg,
            )
            x0_ = self.cfg_denoise_func(self.denoiser2d, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ori_space = self.decoder_motion2d(x0_)  # (B, L, J, 3)
                pred_progress.append(x0_ori_space)  # (B, L, J, 3)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x0_ori_space = self.decoder_motion2d(x)  # (B, L, J, 2)
        outputs["pred_motion2d"] = x0_ori_space  # (B, L, J, 2)
        outputs["pred_motion2d_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, J, 2)
        return outputs


def randomly_set_null_condition(text, f_imgseq, f_camext):
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

    # camera ext
    f_cam_mask = torch.rand(B) < uncond_prob
    f_camext[f_cam_mask] = 0.0

    return text_, f_imgseq, f_camext
