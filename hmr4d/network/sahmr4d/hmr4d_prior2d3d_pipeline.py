import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from einops import einsum
from einops import rearrange, repeat, einsum


from hmr4d.network.gmd.clip import CLIPLatentEncoder
from hmr4d.network.gmd.mdm_unet import MdmUnet
from hmr4d.network.mdm.mdm_net import MDM
from hmr4d.utils.hml3d import convert_bfj3_to_b263f, convert_hmlvec263_to_motion, convert_motion_to_hmlvec263
from hmr4d.utils.pylogger import Log

from hmr4d.utils.geo.proj_constraint import constraint
from hmr4d.utils.geo.triangulation import triangulate_c1v, triangulate_ortho, triangulate_persp, triangulate_2d_3d
from hmr4d.utils.geo_transform import T_transforms_points, project_p2d

import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface

from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
import hmr4d.network.mas.statistics as statistics2d
import hmr4d.network.gmd.statistics as statistics3d

from hmr4d.utils.diffusion.utils import randlike_shape
from hmr4d.network.mas.sample3d_control1v_net import cvt_x_to_c_p2d, cvt_w_p3d_to_x, get_view_noise_for_x
from hmr4d.utils.check_utils import check_equal_get_one


class Hmr4dPrior2d3dPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, args_clip, args_denoiser2d, args_denoiser3d):
        super().__init__()
        self.args = args
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps

        # ----- Prior2D ----- #
        # Scheduler and Denoiser
        self.te_scheduler_2d = self.sch_get_scheduler("ddpm3d", args.scheduler_opt_2d)
        self.clip = CLIPLatentEncoder(**args_clip)
        self.denoiser2d = eval(args_denoiser2d.model)(**args_denoiser2d)

        # Mean, Std
        stats = getattr(statistics2d, self.args.stats_name_2d)
        self.register_buffer("mean_2d", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std_2d", torch.tensor(stats["std"]).float(), False)

        # ----- Prior3D ----- #
        # Scheduler and Denoiser
        self.te_scheduler_3d = self.sch_get_scheduler("ddim", args.scheduler_opt_3d)
        self.denoiser3d = eval(args_denoiser3d.model)(**args_denoiser3d)

        # Mean, Std, and projection
        stats = getattr(statistics3d, self.args.stats_name_3d)
        self.register_buffer("mean_3d", torch.tensor(stats["mean"]).float().reshape(263, 1), False)
        self.register_buffer("std_3d", torch.tensor(stats["std"]).float().reshape(263, 1), False)

        # random projection + scaling
        emph_proj = torch.tensor(statistics3d.GMD_EMPH_PROJ)  # (263, 263)
        self.register_buffer("emph_proj", emph_proj, False)
        self.register_buffer("inv_emph_proj", emph_proj.inverse(), False)

    # ========== Encode / Decode ========== #
    def encode_motion2d(self, motion2d):
        """
        motion2d is in a bbx that in the range of [0, 1]
        this funciton normalize this into standard gaussian distribution
        """
        mean_shape = self.mean_2d.shape  # (J, 2)
        assert motion2d.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        x = (motion2d - self.mean_2d) / self.std_2d
        return x

    def decode_motion2d(self, x):
        """Reverse process of encode_motion2d"""
        mean_shape = self.mean_2d.shape  # (J, 2)
        assert x.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        motion2d = x * self.std_2d + self.mean_2d
        return motion2d

    def encode_hmlvec263(self, hmlvec263):
        """
        HmlVec263 is in the absolute 3D space.
        This function first normalizes it, then uses emphasis-project (proposed by GMD).
        """
        assert hmlvec263.shape[-2] == 263, "Shape should be (B, 263, L)"
        hmlvec263 = (hmlvec263 - self.mean_3d) / self.std_3d
        hmlvec263_proj = einsum(hmlvec263, self.emph_proj, "b d l , d c -> b c l")
        return hmlvec263_proj

    def decode_hmlvec263(self, hmlvec263_proj):
        """Reverse process of encode_hmlvec263"""
        assert hmlvec263_proj.shape[-2] == 263, "Shape should be (B, 263, L)"
        hmlvec263 = einsum(hmlvec263_proj, self.inv_emph_proj, "b d l , d c -> b c l")
        hmlvec263 = hmlvec263 * self.std_3d + self.mean_3d
        return hmlvec263

    # ========== Sample ========== #
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
        enable_cfg = False if self.guidance_scale == 0 else True
        assert enable_cfg == False, "Not implemented yet."
        scheduler_2d = self.te_scheduler_2d
        scheduler_3d = self.te_scheduler_3d

        B = inputs["length"].shape[0]
        L = check_equal_get_one(inputs["length"]) # scalar

        # Setup virtual cameras for 2D diffusion
        V = 3
        Ts_w2c, Ks = get_virtual_cams(B, V, generator.device)
        inputs["Ts_w2c"] = Ts_w2c  # (B, V, 4, 4)
        inputs["Ks"] = Ks  # (B, V, 3, 3)
        noise3d_center = triangulate_ortho(Ts_w2c, torch.zeros((B, V, 1, 2), device=Ks.device))  # (B, 1, 3)
        noise3d_center = noise3d_center[:, None, :, :]  # (B, 1, 1, 3)

        # 1. Prepare target variable x, which will be denoised progressively
        x2d = get_view_noise_for_x((B, L, 22, 3), noise3d_center, Ts_w2c, generator)  # (BV, JC, L)
        x3d = randlike_shape(shape=(B, self.denoiser3d.input_dim, L), generator=generator)

        # 2. Conditions
        text = [""] * B
        prompt_latent = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B or 2B, N_token, 512)
        length_V = repeat(inputs["length"], "b -> (b v)", v=V)
        prompt_latent_V = repeat(prompt_latent, "b n d -> (b v) n d", v=V)

        # DEBUG: providing text-based model with image-based prompt
        # prompt_latent = self.clip.encode_image_sequence(saved_embeds=inputs["saved_embeds"])
        # length_V = repeat(inputs["length"], "b -> (b v)", v=V)
        # prompt_latent_V = repeat(prompt_latent, "b n d -> (b v) n d", v=V)
        # prompt_latent = torch.zeros_like(prompt_latent)  # Do not handle clip_proj in GMD
        # DEBUG: end

        # 3. Prepare scheduler: timestep, extra_step_kwargs
        scheduler_2d.set_timesteps(self.num_inference_steps)
        timesteps_2d = scheduler_2d.timesteps
        extra_step_kwargs_2d = self.prepare_extra_step_kwargs(scheduler_2d, generator)

        scheduler_3d.set_timesteps(self.num_inference_steps)
        timesteps_3d = scheduler_3d.timesteps
        extra_step_kwargs_3d = self.prepare_extra_step_kwargs(scheduler_3d, generator)

        # some sanity check of scheduler
        assert scheduler_2d.order == 1 and scheduler_3d.order == 1

        # 4. Denoising loop
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        for i, (t2d, t3d) in enumerate(zip(timesteps_2d, timesteps_3d)):

            def temp_func1(denoiser, model_kwargs, scheduler, enable_cfg):
                x = model_kwargs.pop("x")
                t = model_kwargs.pop("t")

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

                # a special case since our motion prior is predicting x0
                x0_ = noise_pred
                return x0_

            # == 2D == #
            model_kwargs = dict(x=x2d, t=t2d, prompt_latent=prompt_latent_V, length=length_V)
            x0_2d_ = temp_func1(self.denoiser2d, model_kwargs, scheduler_2d, enable_cfg)
            extra_step_kwargs_2d["view_noise"] = get_view_noise_for_x((B, L, 22, 3), noise3d_center, Ts_w2c, generator)

            # == 3D == #
            model_kwargs = dict(x=x3d, t=t3d, encoder_hidden_states=prompt_latent)
            x0_3d_ = temp_func1(self.denoiser3d, model_kwargs, scheduler_3d, enable_cfg)

            # Do the magic triangulation here, which will overwrite x0_2d_ and x0_3d_
            x0_2d_, x0_3d_, _ = self.triangulate_x0(x0_2d_, x0_3d_, inputs)

            # Update: compute the previous noisy sample x_{t-1} and the original sample x_0
            x2d = scheduler_2d.step(x0_2d_, t2d, x2d, **extra_step_kwargs_2d).prev_sample
            x3d = scheduler_3d.step(x0_3d_, t3d, x3d, **extra_step_kwargs_3d).prev_sample

            # progress bar
            prog_bar.update()

        # Post-processing
        _, _, x_triag = self.triangulate_x0(x2d, x3d, inputs)
        outputs["pred_motion"] = x_triag  # (B, L, 22, 3)

        return outputs

    # ========== Inpaint ========== #
    def triangulate_x0(self, x0_2d_, x0_3d_, inputs):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
            x0_3d_: (B, 263, L)
        """
        method = self.args.triag.method
        opts = self.args.triag.get(method, None)
        if method is None:
            pass
        elif method in ["3d_line_constraint"]:
            x0_3d_, x_triag = self.triag_3d_line_constraint(x0_3d_, inputs, opts)
        elif method in ["2d_line_constraint"]:
            x0_2d_, x_triag = self.triag_2d_line_constraint(x0_2d_, inputs, opts)
        elif method == "2d_triangulation":
            x0_2d_, x_triag = self.triag_2d_triangulation(x0_2d_, inputs, opts)
        elif method == "3d_triangulation":
            x0_3d_, x_triag = self.triag_3d_triangulation(x0_3d_, inputs, opts)
        elif method == "2d_3d_triangulation":
            x0_2d_, x0_3d_, x_triag = self.triag_2d_3d_triangulation(x0_2d_, x0_3d_, inputs, opts)
        elif method == "2dlocal_3d_triangulation":
            x0_2d_, x0_3d_, x_triag = self.triag_2dlocal_3d_triangulation(x0_2d_, x0_3d_, inputs, opts)

        return x0_2d_, x0_3d_, x_triag

    def triag_3d_line_constraint(self, x0_3d_, inputs, opts):
        proj_mode = "persp"
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)

        # Prepare
        pred_hmlvec263 = self.decode_hmlvec263(x0_3d_)  # ( B, 263, L)
        pred_ayf_motion = convert_hmlvec263_to_motion(pred_hmlvec263, abs_3d=True)  # (B, L, 22, 3)
        B, L, J, _ = pred_ayf_motion.shape

        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")

        # Do constraint on pred_ayf_motion
        pred_ayf_motion_ = rearrange(pred_ayf_motion, "b l j c -> b (l j) c")
        pred_ayf_motion_ = constraint(pred_ayf_motion_, T_ayf2c, c_p2d_cv, mode=proj_mode)
        pred_ayf_motion_ = rearrange(pred_ayf_motion_, "b (l j) c -> b l j c", l=L, j=J)
        x_triag = pred_ayf_motion_.clone()

        pred_hmlvec263 = convert_motion_to_hmlvec263(pred_ayf_motion_, return_abs=True)
        x0_inpaint = self.encode_hmlvec263(pred_hmlvec263)

        return x0_inpaint, x_triag

    def triag_2d_line_constraint(self, x0_2d_, inputs, opts):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
        """
        Ts_w2c = inputs["Ts_w2c"]  # (B, V, 4, 4)
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        B, V = Ts_w2c.shape[:2]
        J = 22
        mode = "ortho"
        is_pinhole = mode == "persp"
        bbxs_lurb = repeat(torch.tensor([0, 0, 1, 1], device=Ts_w2c.device), "d -> b v d", b=B, v=V)

        c_p2d = self.decode_motion2d(rearrange(x0_2d_, "bv (j c) l -> bv l j c", j=J, c=2))  # (BV, L, J, 2)
        c_p2d = rearrange(c_p2d, "(b v) l j c -> b v (l j) c", b=B, v=V)
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")

        # triangulate
        w_p3d = triangulate_c1v(Ts_w2c, c_p2d, T_ayf2c, c_p2d_cv, mode=mode, mode_cv="persp")  # (B, LJ, 3)
        x_triag = rearrange(w_p3d, "b (l j) c -> b l j c", j=J, c=3)

        # project to original 2D view
        x0_2d_ = cvt_w_p3d_to_x(w_p3d, Ts_w2c, Ks, bbxs_lurb, is_pinhole, self.encode_motion2d)  # (BV, 22*2, L)
        return x0_2d_, x_triag

    def triag_2d_triangulation(self, x0_2d_, inputs, opts):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
        """
        # obs view
        proj_mode = "persp"
        # proj_mode = "ortho"
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)

        # 2d motion prior
        Ts_w2c = inputs["Ts_w2c"]  # (B, V, 4, 4)
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        B, V = Ts_w2c.shape[:2]
        mode = "ortho"
        is_pinhole = mode == "persp"
        bbxs_lurb = repeat(torch.tensor([0, 0, 1, 1], device=Ts_w2c.device), "d -> b v d", b=B, v=V)

        # triangulate & project to original 2D view
        c_p2d = cvt_x_to_c_p2d(x0_2d_, self.decode_motion2d, bbxs_lurb, Ks)  # (B, V, L*22, 2)

        # Append obs-view to the end
        USE_OBS_VIEW = True
        if USE_OBS_VIEW:
            Ts_w2c_ = torch.cat([Ts_w2c, T_ayf2c[:, None]], dim=1)  # (B, V+1, 4, 4)
            c_p2d_ = torch.cat([c_p2d, c_p2d_cv[:, None]], dim=1)  # (B, V+1, L*22, 2)
            mode_ = [mode] * V + [proj_mode]
            weight_2d_ = [1.0] * V + [1.0]
        else:
            Ts_w2c_, c_p2d_, mode_, weight_2d_ = Ts_w2c, c_p2d, [mode] * V, [1.0] * V
        w_p3d = triangulate_2d_3d(Ts_w2c_, c_p2d_, mode_, w_p3d=None, weight_2d=weight_2d_, weight_3d=None)
        x_triag = rearrange(w_p3d, "b (l j) c -> b l j c", j=22, c=3)
        x0_2d_ = cvt_w_p3d_to_x(w_p3d, Ts_w2c, Ks, bbxs_lurb, is_pinhole, self.encode_motion2d)  # (BV, 22*2, L)
        return x0_2d_, x_triag

    def triag_3d_triangulation(self, x0_3d_, inputs, opts):
        # obs view
        proj_mode = "persp"
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)

        # 3d motion prior
        pred_hmlvec263 = self.decode_hmlvec263(x0_3d_)  # ( B, 263, L)
        pred_ayf_motion = convert_hmlvec263_to_motion(pred_hmlvec263, abs_3d=True)  # (B, L, 22, 3)
        B, L, J, _ = pred_ayf_motion.shape

        # Do constraint on pred_ayf_motion
        pred_ayf_motion_ = rearrange(pred_ayf_motion, "b l j c -> b (l j) c")
        pred_ayf_motion_ = triangulate_2d_3d(
            T_ayf2c[:, None], c_p2d_cv[:, None], proj_mode, pred_ayf_motion_[:, None], weight_2d=1.0
        )
        pred_ayf_motion_ = rearrange(pred_ayf_motion_, "b (l j) c -> b l j c", l=L, j=J)
        x_triag = pred_ayf_motion_.clone()

        pred_hmlvec263 = convert_motion_to_hmlvec263(pred_ayf_motion_, return_abs=True)
        x0_inpaint = self.encode_hmlvec263(pred_hmlvec263)

        return x0_inpaint, x_triag

    def triag_2d_3d_triangulation(self, x0_2d_, x0_3d_, inputs, opts):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
        """
        # obs view
        proj_mode = "persp"
        # proj_mode = "ortho"
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")
        T_ayf2c = inputs["T_ayf2c"]  # (B, 4, 4)

        # 3d motion prior
        pred_hmlvec263 = self.decode_hmlvec263(x0_3d_)  # ( B, 263, L)
        pred_ayf_motion = convert_hmlvec263_to_motion(pred_hmlvec263, abs_3d=True)  # (B, L, 22, 3)
        B, L, J, _ = pred_ayf_motion.shape
        w_p3d_ = rearrange(pred_ayf_motion, "b l j c -> b (l j) c")[:, None]

        # 2d motion prior
        Ts_w2c = inputs["Ts_w2c"]  # (B, V, 4, 4)
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        B, V = Ts_w2c.shape[:2]
        mode = "ortho"
        is_pinhole = mode == "persp"
        bbxs_lurb = repeat(torch.tensor([0, 0, 1, 1], device=Ts_w2c.device), "d -> b v d", b=B, v=V)
        c_p2d = cvt_x_to_c_p2d(x0_2d_, self.decode_motion2d, bbxs_lurb, Ks)  # (B, V, L*22, 2)

        # Append obs-view to the end
        USE_OBS_VIEW = True
        if USE_OBS_VIEW:
            Ts_w2c_ = torch.cat([Ts_w2c, T_ayf2c[:, None]], dim=1)  # (B, V+1, 4, 4)
            c_p2d_ = torch.cat([c_p2d, c_p2d_cv[:, None]], dim=1)  # (B, V+1, L*22, 2)
            mode_ = [mode] * V + [proj_mode]
            weight_2d_ = [1.0] * V + [1.0]
        else:
            Ts_w2c_, c_p2d_, mode_, weight_2d_ = Ts_w2c, c_p2d, [mode] * V, [1.0] * V

        w_p3d = triangulate_2d_3d(Ts_w2c_, c_p2d_, mode_, w_p3d=w_p3d_, weight_2d=weight_2d_, weight_3d=1.0)
        x_triag = rearrange(w_p3d, "b (l j) c -> b l j c", j=22, c=3)

        # Manage output
        x0_2d_ = cvt_w_p3d_to_x(w_p3d, Ts_w2c, Ks, bbxs_lurb, is_pinhole, self.encode_motion2d)  # (BV, 22*2, L)
        pred_hmlvec263 = convert_motion_to_hmlvec263(x_triag.clone(), return_abs=True)
        x0_3d_ = self.encode_hmlvec263(pred_hmlvec263)

        return x0_2d_, x0_3d_, x_triag

    def triag_2dlocal_3d_triangulation(self, x0_2d_, x0_3d_, inputs, opts):
        """
        Args:
            x0_2d_: (BV, 22*2, L)
        """
        # 3d motion prior
        pred_hmlvec263 = self.decode_hmlvec263(x0_3d_)  # ( B, 263, L)
        pred_w_motion = convert_hmlvec263_to_motion(pred_hmlvec263, abs_3d=True)  # (B, L, 22, 3)
        pred_w_root3d = pred_w_motion[:, :, 0, :].unsqueeze(1)  # (B, W=1, L, 3), world is ayf
        B, _, L = x0_3d_.shape[:3]

        # Use input_view + 3D_motion_prior to get global root trajectory
        c_p2d_cv = inputs["c_p2d_cv"]  # (B, L, 22, 2)
        mode_cv = "persp"
        c_root2d_cv = c_p2d_cv[:, :, 0, :].unsqueeze(1)  # (B, V=1, L, 2), root only
        T_ayf2c = inputs["T_ayf2c"].unsqueeze(1)  # (B, V=1, 4, 4)
        w_root3d = triangulate_2d_3d(T_ayf2c, c_root2d_cv, mode_cv, pred_w_root3d)  # (B, L, 3)

        # This trajectory is used to move the centers of virtual cameras
        Ts_w2c = repeat(inputs["Ts_w2c"], "b v c d -> (b l) v c d", l=L).clone()  # (BL, V, 4, 4)
        w_root3d_ = repeat(w_root3d, "b l c -> (b l) v c", v=inputs["Ts_w2c"].shape[1])  # (BL, 3)
        new_t = Ts_w2c[..., :3, 3] - einsum(Ts_w2c[..., :3, :3], w_root3d_, "bl v c d, bl v d -> bl v c")
        Ts_w2c[:, :, :3, 3] = new_t

        # obs view
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> (b l) j c")[:, None]  # (BL, V=1 J, 2)
        T_ayf2c = repeat(T_ayf2c, "b v c d -> (b l) v c d", l=L)  # (BL, V=1, 4, 4)

        # 3d motion prior
        w_p3d_ = rearrange(pred_w_motion, "b l j c -> (b l) j c")[:, None]  # (BL, W=1, 22, 3)

        # 2d motion prior
        Ks = inputs["Ks"]  # (B, V, 3, 3)
        B, V = Ks.shape[:2]
        mode = "ortho"
        is_pinhole = mode == "persp"
        bbxs_lurb = repeat(torch.tensor([0, 0, 1, 1], device=Ks.device), "d -> b v d", b=B, v=V)
        c_p2d = cvt_x_to_c_p2d(x0_2d_, self.decode_motion2d, bbxs_lurb, Ks)  # (B, V, L*22, 2)
        c_p2d = rearrange(c_p2d, "b v (l j) c -> (b l) v j c", j=22)

        # Append obs-view to the end
        USE_OBS_VIEW = True
        if USE_OBS_VIEW:
            Ts_w2c_ = torch.cat([Ts_w2c, T_ayf2c], dim=1)  # (BL, V+1, 4, 4)
            c_p2d_ = torch.cat([c_p2d, c_p2d_cv], dim=1)  # (BL, V+1, 22, 2)
            mode_ = [mode] * V + [mode_cv]
            weight_2d_ = [1.0] * V + [1.0]
        else:
            Ts_w2c_, c_p2d_, mode_, weight_2d_ = Ts_w2c, c_p2d, [mode] * V, [1.0] * V

        w_p3d = triangulate_2d_3d(Ts_w2c_, c_p2d_, mode_, w_p3d=w_p3d_, weight_2d=weight_2d_, weight_3d=1.0)
        x_triag = rearrange(w_p3d, "(b l) j c -> b l j c", b=B)

        # Manage output
        w_p3d_000 = rearrange(x_triag - w_root3d.unsqueeze(2), "b l j c -> b (l j) c")  # (B, L, 22, 3)
        x0_2d_ = cvt_w_p3d_to_x(w_p3d_000, inputs["Ts_w2c"], Ks, bbxs_lurb, is_pinhole, self.encode_motion2d)
        pred_hmlvec263 = convert_motion_to_hmlvec263(x_triag.clone(), return_abs=True)
        x0_3d_ = self.encode_hmlvec263(pred_hmlvec263)

        return x0_2d_, x0_3d_, x_triag


def get_virtual_cams(B, V, device="cpu"):
    """
    Returns:
        Ts_w2c : (B, V, 4, 4)
        Ks : (B, V, 3, 3)
    """

    distance = torch.ones((1,))
    angle = torch.linspace(0, torch.pi, V + 1)[:-1]
    cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle)  # (V, 4, 4)
    Ts_w2c = torch.inverse(cam_mat.to(device))  # (V, 4, 4)
    Ts_w2c = repeat(Ts_w2c, "v c d -> b v c d", b=B)
    Ks = torch.eye(3, device=device).unsqueeze(0).repeat(B, V, 1, 1)  # (B, V, 3, 3)

    return Ts_w2c, Ks
