import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase

import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface

from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.geo.triangulation import triangulate_2d_3d
from hmr4d.utils.geo_transform import apply_T_on_points, transform_mat
from hmr4d.utils.geo_transform import kabsch_algorithm_batch, similarity_transform_batch
from hmr4d.utils.geo.triangulation import triangulate_c1v, triangulate_ortho, triangulate_persp, triangulate_2d_3d
from hmr4d.network.mas.sample3d_control1v_net import cvt_x_to_c_p2d, cvt_w_p3d_to_x, get_view_noise_for_x
from hmr4d.utils.camera_utils import cartesian_to_spherical
from hmr4d.network.clip.clip_text_vision_base_patch32 import CLIPTextOutput


# We use HMR2 feature instead of clip image feature
class Motion2D3DPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser2d, args_denoiser3d, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser2d: denoiser2d network
            args_denoiser3d: denoiser3d network
        """
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        num_visualize = min(args.num_visualize, self.num_inference_steps)
        self.record_interval = self.num_inference_steps // num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Prior2D ----- #
        self.te_scheduler_2d = instantiate(args.scheduler_opt_sample_2d)
        self.denoiser2d = instantiate(args_denoiser2d, _recursive_=False)
        self.data_endecoder2d: EnDecoderBase = instantiate(args.endecoder_opt2d, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder2d.encode
        self.decoder_motion2d = self.data_endecoder2d.decode

        # ----- Prior3D ----- #
        self.te_scheduler_3d = instantiate(args.scheduler_opt_sample_3d)
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        self.data_endecoder3d: EnDecoderBase = instantiate(args.endecoder_opt3d, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder3d.encode
        self.decoder_motion3d = self.data_endecoder3d.decode

        # ----- CLIP Text ----- #
        self.clip = instantiate(args_clip, _recursive_=False)
        self.clip.eval()
        self.clip.requires_grad_(False)

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs_2d(x, timesteps, clip_text, f_imgseq, f_camext, inputs, enable_cfg, **kwargs):
        """override this if you want to add more kwargs"""
        # supermotion/decoder_multiple_crossattn
        V = inputs["V"]
        length_V = repeat(inputs["length"], "b -> (b v)", v=V)
        length_V = torch.cat([length_V] * 2) if enable_cfg else length_V
        f_imgseq_zero = torch.zeros_like(f_imgseq)
        f_imgseq = torch.cat([f_imgseq_zero, f_imgseq]) if enable_cfg else f_imgseq
        f_camext_zero = torch.zeros_like(f_camext)
        f_camext = torch.cat([f_camext_zero, f_camext]) if enable_cfg else f_camext
        model_kwargs = dict(
            x=x,
            timesteps=timesteps,
            length=length_V,
            f_text=clip_text.f_text,
            f_text_length=clip_text.f_text_length,
            f_imgseq=f_imgseq,
            f_camext=f_camext,
        )
        return model_kwargs

    @staticmethod
    def build_model_kwargs_3d(x, timesteps, clip_text, f_imgseq, inputs, enable_cfg, **kwargs):
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
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler_2d = self.te_scheduler_2d
        scheduler_3d = self.te_scheduler_3d

        B = inputs["length"].shape[0]
        max_L = inputs["gt_ayfz_motion"].shape[1]  # B, L, J, 3

        # extra setup: virtual cameras for 2D prior
        inputs["V"] = V = 4
        Ts_w2c, Ks, cam_ext = get_virtual_cams(B, V, generator.device)
        inputs["Ts_w2c"] = Ts_w2c  # (B, V, 4, 4)
        inputs["Ks"] = Ks  # (B, V, 3, 3)

        # 1. Prepare target variable x, which will be denoised progressively
        noise3d_center = triangulate_ortho(Ts_w2c, torch.zeros((B, V, 1, 2), device=Ks.device))  # (B, 1, 3)
        noise3d_center = noise3d_center[:, None, :, :]  # (B, 1, 1, 3)
        x2d = get_view_noise_for_x((B, max_L, 22, 3), noise3d_center, Ts_w2c, generator)  # (BV, JC, L)
        x3d = randlike_shape(shape=(B, self.denoiser3d.input_dim, max_L), generator=generator)

        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]
        if not enable_cfg:
            text = ["" for _ in range(len(text))]
            f_imgseq = torch.zeros_like(f_imgseq)

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B, 77, D)
        if enable_cfg:
            f_text_uncond, f_text_cond = clip_text.f_text.chunk(2)
            f_text_V = torch.cat(
                [repeat(f_text_uncond, "b l d -> (b v) l d", v=V), repeat(f_text_cond, "b l d -> (b v) l d", v=V)]
            )

            f_text_length_uncond, f_text_length_cond = clip_text.f_text_length.chunk(2)
            f_text_length_V = torch.cat(
                [repeat(f_text_length_uncond, "b -> (b v)", v=V), repeat(f_text_length_cond, "b -> (b v)", v=V)]
            )
        else:
            f_text_V = repeat(clip_text.f_text, "b l d -> (b v) l d", v=V)
            f_text_length_V = repeat(clip_text.f_text_length, "b -> (b v)", v=V)
        clip_text_V = CLIPTextOutput(f_text_V, f_text_length_V)
        f_camext_V = rearrange(cam_ext, "b v c -> (b v) 1 c")  # (BV, 1, 4)
        f_imgseq_V = repeat(f_imgseq, "b l c -> (b v) l c", v=V)  # (BV, L, 1024)

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler_2d.set_timesteps(self.num_inference_steps)
        timesteps_2d = scheduler_2d.timesteps
        extra_step_kwargs_2d = self.prepare_extra_step_kwargs(scheduler_2d, generator)

        scheduler_3d.set_timesteps(self.num_inference_steps)
        timesteps_3d = scheduler_3d.timesteps
        extra_step_kwargs_3d = self.prepare_extra_step_kwargs(scheduler_3d, generator)

        # some sanity check of scheduler
        assert scheduler_2d.order == 1 and scheduler_3d.order == 1
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        pred_progress = []  # for visualization
        pred_2d_progress = []
        w2c_progress = []
        for i, (t2d, t3d) in enumerate(zip(timesteps_2d, timesteps_3d)):
            # == 2D == #
            model_kwargs = self.build_model_kwargs_2d(
                x=x2d,
                timesteps=t2d,
                clip_text=clip_text_V,
                f_imgseq=f_imgseq_V,
                f_camext=f_camext_V,
                inputs=inputs,
                enable_cfg=enable_cfg,
            )
            x0_2d_ = self.cfg_denoise_func(self.denoiser2d, model_kwargs, scheduler_2d, enable_cfg)
            extra_step_kwargs_2d["view_noise"] = get_view_noise_for_x(
                (B, max_L, 22, 3), noise3d_center, Ts_w2c, generator
            )

            # == 3D == #
            model_kwargs = self.build_model_kwargs_3d(
                x=x3d,
                timesteps=t3d,
                clip_text=clip_text,
                f_imgseq=f_imgseq,
                inputs=inputs,
                enable_cfg=enable_cfg,
            )
            x0_3d_ = self.cfg_denoise_func(self.denoiser3d, model_kwargs, scheduler_3d, enable_cfg)

            # Do the magic triangulation here, which will overwrite x0_2d_ and x0_3d_
            # weight_2d = 1.0 if i > 50 else 0.0
            weight_2d = 1.0
            x0_2d_, x0_3d_, _, Ts_w2c_follow = self.triangulate_x0(x0_2d_, x0_3d_, inputs, weight_2d=weight_2d)

            # Update: compute the previous noisy sample x_{t-1} and the original sample x_0
            x2d = scheduler_2d.step(x0_2d_, t2d, x2d, **extra_step_kwargs_2d).prev_sample
            x3d = scheduler_3d.step(x0_3d_, t3d, x3d, **extra_step_kwargs_3d).prev_sample

            # *. Update and store intermediate results
            if i % self.record_interval == 0:
                pred_progress.append(self.decoder_motion3d(x3d))  # (B, L, J, 3)
                pred_2d_progress.append(
                    rearrange(self.decoder_motion2d(x2d), "(b v) l j c -> b v l j c", b=B)
                )  # (B, V, L, J, 2)
                w2c_progress.append(Ts_w2c_follow)  # (B, L, V, 4, 4)

            # progress bar
            prog_bar.update()

        # Post-processing
        _, _, x_triag, _ = self.triangulate_x0(x2d, x3d, inputs)
        outputs["pred_ayfz_motion"] = x_triag  # (B, L, J, 3)
        outputs["pred_ayfz_motion_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, J, 3)
        outputs["pred_motion_2d_progress"] = torch.stack(pred_2d_progress, dim=1)  # (B, Progress, V, L, J, 2)
        outputs["pred_w2c_progress"] = torch.stack(w2c_progress, dim=1)  # (B, Progress, L, V, 4, 4)
        return outputs

    # ========== Inpaint ========== #
    def triangulate_x0(self, x0_2d_, x0_3d_, inputs, weight_2d=1.0):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
            x0_3d_: (B, 263, L)
        """
        # x_triag = self.decoder_motion3d(x0_3d_)  # (B, L, J, 3)
        # return x0_2d_, x0_3d_, x_triag

        # 3d motion prior
        B, _, L = x0_3d_.shape[:3]
        pred_w_motion = self.decoder_motion3d(x0_3d_)  # (B, L, J, 3)
        pred_w_root3d = pred_w_motion[:, :, 0, :].unsqueeze(1)  # (B, W=1, L, 3), world is ayfz

        # Use input_view + 3D_motion_prior to get global root trajectory
        # c_p2d_cv = inputs["c_p2d_cv"]  # (B, L, 22, 2)
        # mode_cv = "persp"
        # c_root2d_cv = c_p2d_cv[:, :, 0, :].unsqueeze(1)  # (B, V=1, L, 2), root only
        # T_ayf2c = inputs["T_ayf2c"].unsqueeze(1)  # (B, V=1, 4, 4)
        # w_root3d = triangulate_2d_3d(T_ayf2c, c_root2d_cv, mode_cv, pred_w_root3d)  # (B, L, 3)
        w_root3d = pred_w_root3d.reshape(B, L, 3)  # (B, L, 3)

        # This trajectory is used to move the centers of virtual cameras
        Ts_w2c = repeat(inputs["Ts_w2c"], "b v c d -> (b l) v c d", l=L).clone()  # (BL, V, 4, 4)
        w_root3d_ = repeat(w_root3d, "b l c -> (b l) v c", v=inputs["Ts_w2c"].shape[1])  # (BL, 3)
        new_t = Ts_w2c[..., :3, 3] - einsum(Ts_w2c[..., :3, :3], w_root3d_, "bl v c d, bl v d -> bl v c")
        Ts_w2c[:, :, :3, 3] = new_t

        # obs view
        # c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> (b l) j c")[:, None]  # (BL, V=1 J, 2)
        # T_ayf2c = repeat(T_ayf2c, "b v c d -> (b l) v c d", l=L)  # (BL, V=1, 4, 4)

        # 3d motion prior
        w_p3d_ = rearrange(pred_w_motion, "b l j c -> (b l) j c")[:, None]  # (BL, W=1, 22, 3)

        # 2d motion prior
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        B, V = Ks.shape[:2]
        mode = "ortho"
        is_pinhole = mode == "persp"
        bbxs_lurb = repeat(torch.tensor([0, 0, 1, 1], device=Ks.device), "d -> b v d", b=B, v=V)
        # c_p2d = cvt_x_to_c_p2d(x0_2d_, self.decoder_motion2d, bbxs_lurb, Ks)  # (B, V, L*22, 2)
        # c_p2d = rearrange(c_p2d, "b v (l j) c -> (b l) v j c", j=22)
        c_p2d = self.decoder_motion2d(x0_2d_)  # (BV, L, 22, 2)
        c_p2d = rearrange(c_p2d, "(b v) l j c -> (b l) v j c", b=B)

        # Append obs-view to the end
        USE_OBS_VIEW = False
        if USE_OBS_VIEW:
            Ts_w2c_ = torch.cat([Ts_w2c, T_ayf2c], dim=1)  # (BL, V+1, 4, 4)
            c_p2d_ = torch.cat([c_p2d, c_p2d_cv], dim=1)  # (BL, V+1, 22, 2)
            mode_ = [mode] * V + [mode_cv]
            weight_2d_ = [1.0] * V + [1.0]
        else:
            Ts_w2c_, c_p2d_, mode_, weight_2d_ = Ts_w2c, c_p2d, [mode] * V, [weight_2d] * V

        w_p3d = triangulate_2d_3d(Ts_w2c_, c_p2d_, mode_, w_p3d=w_p3d_, weight_2d=weight_2d)
        x_triag = rearrange(w_p3d, "(b l) j c -> b l j c", b=B)

        # Manage output
        w_p3d_000 = rearrange(x_triag - w_root3d.unsqueeze(2), "b l j c -> b (l j) c")  # (B, L, 22, 3)

        x0_2d_ = cvt_w_p3d_to_x(w_p3d_000, inputs["Ts_w2c"], Ks, bbxs_lurb, is_pinhole, self.encoder_motion2d)
        x0_3d_ = self.encoder_motion3d(x_triag)

        Ts_w2c_ = rearrange(Ts_w2c, "(b l) v c d -> b l v c d", b=B)

        return x0_2d_, x0_3d_, x_triag, Ts_w2c_


def get_virtual_cams(B, V, device="cpu"):
    """
    Returns:
        Ts_w2c : (B, V, 4, 4)
        Ks : (B, V, 3, 3)
        cam_ext : (B, V, 4)
    """

    distance = torch.ones((1,))
    angle = torch.linspace(0, 2 * torch.pi, V + 1)[:-1]  # Start = 0
    cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle)  # (V, 4, 4)
    cam_mat = cam_mat.to(device)

    # Create output
    Ts_w2c = torch.inverse(cam_mat)  # (V, 4, 4)
    Ts_w2c = repeat(Ts_w2c, "v c d -> b v c d", b=B)
    Ks = torch.eye(3, device=device).unsqueeze(0).repeat(B, V, 1, 1)  # (B, V, 3, 3)

    spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat))  # (V, 3)
    theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]
    cam_ext = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=-1)  # (V, 4)
    cam_ext = repeat(cam_ext, "v c -> b v c", b=B)  # (B, V, 4)

    return Ts_w2c, Ks, cam_ext
