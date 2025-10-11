import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers import DDIMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.mas.utils.motion2d_endecoder import EnDecoderBase

import time
from hmr4d.utils.geo.triangulation import triangulate_ortho, triangulate_persp
from hmr4d.dataset.motionx.utils import normalize_keypoints_to_patch, normalize_kp_2d
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_from_bi01_p2d,
    cvt_p2d_from_i_to_c,
)
from hmr4d.model.mas.pipeline.mas import localmotion_withnr_to_globalmotion
from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.plt_utils import plt_skeleton_animation, visualize_2d_motion_as_video_mv

class MVPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoisermv, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoisermv: denoisermv network
        """
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.record_interval = self.num_inference_steps // args.num_visualize if args.num_visualize > 0 else torch.inf

        # ----- Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = instantiate(args.scheduler_opt_sample)
        self.te3d_scheduler = instantiate(args.scheduler_opt_sample3d)

        # ----- Networks ----- #
        self.clip = instantiate(args_clip, _recursive_=False)
        args_denoisermv["with_projection"] = args["with_projection"]
        self.denoisermv = instantiate(args_denoisermv, _recursive_=False)
        Log.info(self.denoisermv)

        if self.args.noise2d_scale > 0:
            Log.info(f"Use noise scale = {self.args.noise2d_scale} on 2d inputs")

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_motion2d = self.data_endecoder.encode
        self.decoder_motion2d = self.data_endecoder.decode
        
        # smplx neutral beta=0
        # lowerbody 1.0402, upperbody: 0.6385

        # self._iter = 0
        # self._total_time = 0.
        
        # total_params = sum(p.numel() for p in self.denoisermv.parameters())
        # print(f"Total param: {total_params /1024 / 1024:2f}M")

        self.neutral_height = 1.6787
        self.neutral_bone_length = [
            0.1125,
            0.3828,
            0.4066,
            0.1399,
            0.1235,
            0.3655,
            0.4115,
            0.1381,
            0.1133,
            0.1323,
            0.0605,
            0.1686,
            0.1637,
            0.0978,
            0.1164,
            0.2749,
            0.2498,
            0.0970,
            0.1334,
            0.2676,
            0.2531,
        ]
        # ----- Freeze ----- #
        self.freeze_clip()

        # ----- Freeze 2D diffusion ----- #
        if self.args.is_freeze2d:
            self.freeze_denoiser2d()
        self.i = 0

    def freeze_clip(self):
        Log.info("Freezing CLIP")
        self.clip.eval()
        self.clip.requires_grad_(False)

    def freeze_denoiser2d(self):
        Log.info("Freezing 2D denoiser")
        self.denoisermv.freeze()

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs(x, timesteps, length, f_condition, enable_cfg=False):
        if enable_cfg:
            length = torch.cat([length, length])
            for k in f_condition.keys():
                if k == "f_text":
                    pass
                else:
                    f_condition[k] = torch.cat([torch.zeros_like(f_condition[k]), f_condition[k]])
        return dict(x=x, timesteps=timesteps, length=length, **f_condition)

    def forward_train(self, inputs):
        #Log.info("pipeline.mv: forward_train")
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        motion = inputs["gt_motion2d"]  # (B, V, L, J, 3)
        T_w2c = inputs["T_w2c"]  # (B, V, 4, 4)
        scheduler = self.tr_scheduler
        B, V, L, J, _ = motion.shape

        # *. Encoding
        x = self.encoder_motion2d(motion)  # (B, V, C, L)

        # *. Add noise
        noise, _ = get_view_noise((B, V, J, L), T_w2c, self.args.is_worldnoise)
        # t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,)).long().to(x.device)
        noisy_x = scheduler.add_noise(x, noise, t)

        # Conditions
        assert self.training
        f_condition = {
            "f_text": inputs["text"],  # (B, C)
            "f_cam": inputs["cam_emb"],  # (B, V, C)
        }
        if self.args.is_2dinput:
            f_cond_2d = x[:, 0]  # (B, C, L)
            if self.args.noise2d_scale > 0:
                f_cond_2d = f_cond_2d + torch.randn_like(f_cond_2d) * self.args.noise2d_scale
            f_condition["f_cond2d"] = f_cond_2d  # (B, C, L)

        f_condition = randomly_set_null_condition(f_condition, self.args.is_hastext)
        clip_text = self.clip.encode_text(
            f_condition["f_text"], enable_cfg=False, with_projection=self.args.with_projection
        )
        f_condition["f_text"] = clip_text.f_text  # (B, D)

        # Forward
        model_kwargs = self.build_model_kwargs(x=noisy_x, timesteps=t, length=length, f_condition=f_condition)
        model_kwargs["f_text_length"] = clip_text.f_text_length

        model_output = self.denoisermv(**model_kwargs)
        model_pred = model_output.sample  # (B, V, C, L)
        mask = model_output.mask  # (B, 1, 1, L)

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

        outputs["simple_loss"] = loss
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
        return x0_, denoiser_out

    # ========== Sample 2D ========== #
    def forward_sample(self, inputs):
        print("Multi-view generation uses forward_sample3d")
        raise NotImplementedError

    # ========== Sample 3D ========== #
    def forward_sample3d(self, inputs):
        """Generate 3D Motion (22 joints) (B, L, 22 * 3)"""     
        left_leg_pair = [
            [0, 1],
            [1, 4],
            [4, 7],
            [7, 10],
        ]
        right_leg_pair = [
            [0, 2],
            [2, 5],
            [5, 8],
            [8, 11],
        ]
        spine_pair = [
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
        ]

        joint_pair = [
            [0, 1],
            [1, 4],
            [4, 7],
            [7, 10],
            [0, 2],
            [2, 5],
            [5, 8],
            [8, 11],
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            [9, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            [9, 13],
            [13, 16],
            [16, 18],
            [18, 20],
        ]

        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.te3d_scheduler

        length = inputs["length"]
        B = inputs["length"].shape[0]
        max_L = length.max().item()

        T_w2c = inputs["T_w2c"]  # (B, V, 4, 4)
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        # is_pinhole = inputs["is_pinhole"][0].item()
        is_pinhole = self.args.is_perspective
        patch_size = inputs["patch_size"][0].item()
        bbox_scale = inputs["cam_emb"][:, 0, -1] # (B, )
        B, V = T_w2c.shape[:2]
        BV = B * V
        BL = B * max_L
        BVL = B * V * max_L
        J = self.data_endecoder.J

        triangulate_func = triangulate_persp if self.args.is_perspective else triangulate_ortho

        # Functions
        def cvt_x_to_c_p2d(x):
            try:
                normed_p2d = self.decoder_motion2d(x)  # (B, V, L, J, 2)
            except RuntimeError:
                # separate process zero root
                normed_p2d = self.decoder_motion2d(x[..., 2:, :])  # (B, V, L, J, 2)
                normed_p2d = torch.cat(
                    [torch.zeros_like(normed_p2d[..., :1, :]), normed_p2d], dim=-2
                )  # (B, V, L, J, 2)

            if self.args.is_perspective:
                for b_i in range(B):
                    normed_p2d[b_i] = normed_p2d[b_i] * bbox_scale[b_i] * 200. / 224.

                normed_p2d = rearrange(normed_p2d, "b v l j c -> (b v l) j c", b=B, v=V, l=max_L)  # (BVL, J, 2)
                i_p2d = normalize_keypoints_to_patch(normed_p2d, crop_size=patch_size, inv=True)  # (BVL, J, 2)
                c_p2d = cvt_p2d_from_i_to_c(i_p2d, repeat(Ks, "b v c d -> (b v l) c d", l=max_L))  # (BVL, J, 2)
            else:
                i_p2d = rearrange(normed_p2d, "b v l j c -> (b v l) j c", b=B, v=V, l=max_L)  # (BVL, J, 2)
                c_p2d = i_p2d

            c_p2d = rearrange(c_p2d, "(b v l) j c -> b v (l j) c", b=B, v=V, l=max_L)  # (B, V, L*J, 2)
            return c_p2d

        def cvt_w_p3d_to_x(w_p3d):
            c_p3d = apply_T_on_points(
                repeat(w_p3d, "b (l j) c -> (b v l) j c", v=V, l=max_L),
                repeat(T_w2c, "b v c d -> (b v l) c d", l=max_L),
            )  # (BVL, J, 3)
            
            if self.args.is_perspective:
                i_p2d = project_p2d(
                    c_p3d, repeat(Ks, "b v c d -> (b v l) c d", l=max_L), is_pinhole=is_pinhole
                )  # (BVL, J, 2)
                ############## way1, assume is already 224, but not right
                normed_p2d = normalize_keypoints_to_patch(i_p2d, crop_size=patch_size)  # (BVL, J, 2)
                normed_p2d = rearrange(normed_p2d, "(b v l) j c -> b v l j c", b=B, v=V, l=max_L)
                ###

                ### way2, assume is not 224, use input scale
                i_p2d = rearrange(i_p2d, "(b v l) j c -> b v l j c", b=B, v=V, l=max_L)
                for b_i in range(B):
                    l_i = length[b_i]
                    i_p2d_i = i_p2d[b_i, :, :l_i].detach().cpu()
                    # normed_p2d_i, bbox_p2d_i, bbox_i = normalize_kp_2d(i_p2d_i, not_moving=True, multiview=True, randselect=False)
                    normed_p2d_i, bbox_p2d_i, bbox_i = normalize_kp_2d(i_p2d_i, not_moving=True, multiview=True, randselect=False, scale=bbox_scale[b_i].item())
                    normed_p2d[b_i, :, :l_i] = normed_p2d_i.to(i_p2d.device)
                ##############
            else:
                normed_p2d = c_p3d[..., :2] # (BVL, J, 2)
                normed_p2d = rearrange(normed_p2d, "(b v l) j c -> b v l j c", b=B, v=V, l=max_L)  # (B, V, L, J, 2)

            try:
                x = self.encoder_motion2d(normed_p2d)  # (B, V, J*2, L)
            except RuntimeError:
                # Separate process zero root
                normed_p2d = normed_p2d[..., 1:, :]  # (B, V, L, J, 2)
                x = self.encoder_motion2d(normed_p2d)  # (B, V, J*2, L)
                x = torch.cat([torch.zeros_like(x[..., :2, :]), x], dim=-2)  # (B, V, J*2, L)
            return x  # (BV, J*2, L)

        def cvt_w_p3d_to_c_p2d(w_p3d):
            c_p3d = apply_T_on_points(
                repeat(w_p3d, "b (l j) c -> (b v l) j c", v=V, l=max_L), repeat(T_w2c, "b v c d -> (b v l) c d", l=L)
            )  # (BVL, J, 3)
            c_p2d = rearrange(c_p3d[..., [0, 1]], "(b v l) j c -> b v (l j) c", b=B, v=V, l=max_L)  # (B, V, L*J, 2)
            return c_p2d
        
        # start_time = time.time()

        # 1. Prepare target variable x, which will be denoised progressively
        x, noise3d = get_view_noise(
            (B, V, J, max_L), T_w2c, self.args.is_worldnoise, generator
        )  # in the data space after normalization
        # (B, V, J*2, L)

        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]

        # assign text with zero when no cfg
        if not enable_cfg:
            text = ["" for _ in range(len(text))]

        clip_text = self.clip.encode_text(
            text, enable_cfg=enable_cfg, with_projection=self.args.with_projection
        )
        f_text = clip_text.f_text

        assert not self.args.is_2dinput

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator)  # for scheduler.step()
        pred_progress = {}  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            f_condition = {
                "f_text": f_text,  # (B, D)
                "f_cam": inputs["cam_emb"],  # (B, V, C)
            }
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                length=length,
                f_condition=f_condition,
                enable_cfg=enable_cfg,
            )
            model_kwargs["f_text_length"] = clip_text.f_text_length

            x0_, denoiser_out = self.cfg_denoise_func(self.denoisermv, model_kwargs, scheduler, enable_cfg)
            # (B, V, J*2, L)

            view_noise, _ = get_view_noise(
                (B, V, J, max_L), T_w2c, self.args.is_worldnoise, generator
            )  # (B, V, J*2, L)
            extra_step_kwargs["view_noise"] = view_noise  # (B, V, J*2, L)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample


            if self.args.is_consisblock or i == len(timesteps) - 1:
                # root is always zero, add zero root
                x0_ = torch.cat([torch.zeros_like(x0_[..., :2, :]), x0_], dim=-2)

                # triangulation
                c_p2d = cvt_x_to_c_p2d(x0_)  # (B, V, L*J, 2)
                w_p3d = triangulate_func(T_w2c, c_p2d)  # (B, L*J, 3)
            
            if self.args.is_lastnorm and i == len(timesteps) - 1:
                # normalize to neutral height
                if self.args.is_heightnorm:
                    w_p3d = rearrange(w_p3d, "b (l j) c -> b l j c", l=max_L, c=3)
                    left_leg_length = 0
                    for pair in left_leg_pair:
                        left_leg_length += (w_p3d[:, :, pair[0], :] - w_p3d[:, :, pair[1], :]).norm(dim=-1)
                    right_leg_length = 0
                    for pair in right_leg_pair:
                        right_leg_length += (w_p3d[:, :, pair[0], :] - w_p3d[:, :, pair[1], :]).norm(dim=-1)
                    lower_body_length = (left_leg_length + right_leg_length) / 2
                    spine_length = 0
                    for pair in spine_pair:
                        spine_length += (w_p3d[:, :, pair[0], :] - w_p3d[:, :, pair[1], :]).norm(dim=-1)
                    height = lower_body_length + spine_length
                    scale = self.neutral_height / (height + 1e-9)  # (B, L)
                    scale = torch.clamp(scale, min=0.0, max=5.0)
                    w_p3d = w_p3d * scale[..., None, None]  # for nr scale
                    w_p3d = rearrange(w_p3d, "b l j c -> b (l j) c")
                ##################

            if self.args.is_consisblock or i == len(timesteps) - 1:
                # for vis
                w_p3d_ = rearrange(w_p3d, "b (l j) c -> b l j c", l=max_L, c=3)  # (B, L, J, 3)
                c_p2d_ = rearrange(c_p2d, "b v (l j) c -> b v l j c", v=V, l=max_L, c=2)  # (B, V, L, J, 2)

            # *. Update x0_
            if self.args.is_consisblock:
                # Project x0: (B, V, J*2, L)
                x0_ = cvt_w_p3d_to_x(w_p3d)  # (B, V, J*2, L)
                # Remove zero root
                x0_ = x0_[..., 2:, :]

                # Use posterior p(x{t-1} | xt, x0) with projected x0
                scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
                x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            if i % self.record_interval == 0:
                if "pred_motion" not in pred_progress.keys():
                    pred_progress["pred_motion"] = []
                    pred_progress["pred_motion2d"] = []
                if self.args.is_consisblock or i == len(timesteps) - 1:
                    pred_progress["pred_motion"].append(w_p3d_)  # (B, L, J, 3)
                    pred_progress["pred_motion2d"].append(c_p2d_)  # (B, V, L, J, 2)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        plt_2d = c_p2d_[0, :, :, :, :].squeeze(0)
        # print(f"2d: {plt_2d.shape}")
        # visualize_2d_motion_as_video_mv(plt_2d, text[0], index = self.i)
        self.i += 1
        # plt_skeleton_animation(plt_2d, skeleton_type="smpl")
        # plt_skeleton_animation_mv(plt_2d, skeleton_type="smpl")
        outputs["pred_motion"] = w_p3d_  # (B, L, J, 3)
        outputs["pred_global_motion"] = localmotion_withnr_to_globalmotion(w_p3d_)  # (B, L, J, 3)
        # end_time = time.time()
        # self._iter += 1

        # if self._iter > 10:
        #     self._total_time += (end_time - start_time)
        # if self._iter == 20:
        #     avg_time = self._total_time / 10
        #     print(f"Average time: {avg_time}")
        #     raise NotImplementedError

        for k in pred_progress.keys():
            pred_progress[k] = torch.stack(pred_progress[k], dim=1)  # (B, Progress, L, J, 3)
        outputs["pred_progress"] = pred_progress
        return outputs


def randomly_set_null_condition(f_condition, is_hastext, uncond_prob=0.1):
    """
    To support classifier-free guidance, randomly set-to-unconditioned,
    Conditions are in shape (B, L, C)
    f_text: List of str
    """
    B = len(f_condition["f_text"])

    keys = list(f_condition.keys())
    for k in keys:
        if k == "f_text":
            # text
            text = f_condition[k]
            text_mask = torch.rand(B) < uncond_prob
            if is_hastext:
                text_ = ["" if m else t for m, t in zip(text_mask, text)]
            else:
                text_ = [""] * B
            f_condition[k] = text_
        else:
            f_condition[k] = f_condition[k].clone()
            mask = torch.rand(B, f_condition[k].shape[1]) < uncond_prob
            f_condition[k][mask] = 0.0

    return f_condition


def get_view_noise(noise_shape, T_w2c, is_worldnoise=True, generator=None):
    """Get view noise (BV, J*2, L)
    Args:
        is_worldnoise: if True, the noise is multiview-consistent
    """
    B, V, J, max_L = noise_shape
    if is_worldnoise:
        if generator is not None:
            noise3d = randlike_shape((B, max_L * J, 3), generator)
            # noise3d = randlike_shape((B, max_L * J, 3), generator)
        else:
            noise3d = torch.randn((B, max_L * J, 3), device=T_w2c.device)
        noise3d = torch.randn((B, max_L * J, 3)).to(T_w2c.device)
        view_noise3d = apply_T_on_points(  # (BVL, J, 3)
            repeat(noise3d, "b (l j) c -> (b v l) j c", v=V, l=max_L),
            repeat(T_w2c, "b v c d -> (b v l) c d", l=max_L),
        )
        view_noise = view_noise3d[..., [0, 1]]
        view_noise = rearrange(view_noise, "(b v l) j c -> b v (j c) l", b=B, v=V, l=max_L)
    else:
        if generator is not None:
            view_noise = randlike_shape((B, V, J * 2, max_L), generator)
            noise3d = randlike_shape((B, max_L * J, 3), generator)
        else:
            view_noise = torch.randn((B, V, J * 2, max_L), device=T_w2c.device)
            noise3d = torch.randn((B, max_L * J, 3), device=T_w2c.device)

        view_noise = torch.randn((B, V, J * 2, max_L)).to(T_w2c.device)
        noise3d = torch.randn((B, max_L * J, 3)).to(T_w2c.device)
    return view_noise, noise3d  # view_noise is (B, V, J*2, L), noise3d is (B, L*J, 3)
