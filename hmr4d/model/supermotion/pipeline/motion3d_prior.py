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
from hmr4d.utils.geo.optim import find_wst_transform, update_pred_with_wst
from hmr4d.utils.geo_transform import apply_T_on_points, transform_mat
from hmr4d.utils.geo_transform import kabsch_algorithm_batch, similarity_transform_batch
from hmr4d.utils.filter import smooth_pose_oneeurofilter, smooth_pose_savgol, smooth_pose_gaussian
from pytorch3d.ops import efficient_pnp
from hmr4d.utils.geo_transform import ransac_PnP_batch, transform_mat
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from scipy.ndimage._filters import _gaussian_kernel1d
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


# We use HMR2 feature instead of clip image feature
class Motion3DPriorPipeline(nn.Module, PipelineHelper):
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
        self.data_endecoder: EnDecoderBase = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode
        self.decoder_motion3d = self.data_endecoder.decode

        # ----- Freeze ----- #
        self.freeze_clip()

        # -- temporal variable -- #
        self.T_ayfz2c = None
        sigma = 5
        kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
        kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None]  # (1, 1, K)
        self.register_buffer("kernel_smooth", kernel_smooth, persistent=False)

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
        motion = inputs["gt_ayfz_motion"]  # (B, L, J, 3)
        length = inputs["length"]  # (B,) effective length of each sample
        scheduler = self.tr_scheduler
        B, L, J, _ = motion.shape

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

    # ========== Sample ========== #
    def forward_sample(self, inputs):
        """Generate Motion (22 joints) (B, L, 263)"""
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.te_scheduler

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

        max_L = inputs["gt_ayfz_motion"].shape[1]  # B, L, J, 3

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.denoiser3d.input_dim, max_L), generator=generator)

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

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_
            if i % self.record_interval == 0:
                x0_ori_space = self.decoder_motion3d(x0_)  # (B, L, J, 3)
                pred_progress.append(x0_ori_space)  # (B, L, J, 3)

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        # Post-processing
        x0_ori_space = self.decoder_motion3d(x)  # (B, L, J, 3)
        outputs["pred_ayfz_motion"] = x0_ori_space  # (B, L, J, 3)
        outputs["pred_ayfz_motion_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, J, 3)
        return outputs

    # ========== Mocap ========== #

    def forward_mocap(self, inputs):
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler_3d = self.te_scheduler

        length = inputs["length"]  # (B,) effective length of each sample
        B = inputs["length"].shape[0]
        L = inputs.get("obs_c_p2d", inputs.get("obs_cr_p3d")).shape[1]  # must provide 2D or 3D observation
        assert L == 300

        # 1. Prepare target variable x, which will be denoised progressively
        x3d = randlike_shape(shape=(B, self.denoiser3d.input_dim, L), generator=generator)

        # 2. Conditions
        # Encode CLIP embedding
        text = inputs["text"]
        f_imgseq = inputs["f_imgseq"]

        # fitler 2d pose
        obs_c_p2d = inputs["obs_c_p2d"]

        obs_c_p2d_max = obs_c_p2d.reshape(B, -1, 2).max(dim=1)[0]  # B, 2
        obs_c_p2d_min = obs_c_p2d.reshape(B, -1, 2).min(dim=1)[0]  # B, 2
        obs_c_p2d_scale = (obs_c_p2d_max - obs_c_p2d_min).norm(dim=-1)  # B

        vel_2d = obs_c_p2d[:, 1:] - obs_c_p2d[:, :-1]  # B, L - 1, J, 2
        vel_2d = vel_2d.norm(dim=-1).max(dim=-1)[0]  # B, L - 1
        vel_2d = torch.cat([vel_2d, torch.zeros((B, 1), device=vel_2d.device)], dim=-1)  # B, L

        vel_2d = vel_2d / rearrange(obs_c_p2d_scale, "b -> b 1").clamp(min=1e-5)

        jitter_mask = vel_2d > 0.1

        # filter noisy 2d obs
        # obs_c_p2d = smooth_pose_oneeurofilter(obs_c_p2d, min_cutof=0.004, beta=1.5, d_cutof=0.004)
        # obs_c_p2d = smooth_pose_savgol(obs_c_p2d, window_length=15, polyorder=3)
        # obs_c_p2d = smooth_pose_gaussian(obs_c_p2d, sigma=3)
        inputs["obs_c_p2d"] = obs_c_p2d

        # make jitter mask
        conv = nn.Conv1d(1, 1, 3, 1, padding=1, bias=False).to(obs_c_p2d.device)
        nn.init.ones_(conv.weight)
        jitter_mask = conv(jitter_mask[:, None].float())[:, 0] > 0
        inputs["jitter_mask"] = jitter_mask

        if not enable_cfg:
            text = ["" for _ in range(len(text))]
            f_imgseq = torch.zeros_like(f_imgseq)
        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg)  # (B, 77, D)

        # 3. Prepare scheduler: timestep, extra_step_kwargs
        scheduler_3d.set_timesteps(self.num_inference_steps)
        timesteps_3d = scheduler_3d.timesteps
        extra_step_kwargs_3d = self.prepare_extra_step_kwargs(scheduler_3d, generator)

        # some sanity check of scheduler
        assert scheduler_3d.order == 1

        # 5. Get smooth mask for obs
        obs_cr_p3d = inputs.get("obs_cr_p3d", None)  # (B, L, 22, 3)
        if obs_cr_p3d is not None:
            smooth_obs_cr_p3d = self.make_smooth(obs_cr_p3d, dim=1, length=length)
            # obs_cr_p3d = smooth_obs_cr_p3d
        obs_c_p2d = inputs.get("obs_c_p2d", None)  # (B, L, 22, 2)
        if obs_c_p2d is not None:
            smooth_obs_c_p2d = self.make_smooth(obs_c_p2d, dim=1, length=length)
            # obs_c_p2d = smooth_obs_c_p2d
        T_ayfz2c = inputs.get("T_ayfz2c", None)  # (B, 1, 4, 4)
        if T_ayfz2c is not None:  # Use a predefined T_ayfz2c (utility for sliding window inference)
            self.T_ayfz2c = T_ayfz2c

        # 4. Denoising loop
        pred_progress = []
        prior_progress = []
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        for i, t3d in enumerate(timesteps_3d):
            # == 3D == #
            model_kwargs = self.build_model_kwargs(
                x=x3d,
                timesteps=t3d,
                clip_text=clip_text,
                f_imgseq=f_imgseq,  # if i < 50 else torch.zeros_like(f_imgseq),
                inputs=inputs,
                enable_cfg=enable_cfg,
            )
            x0_3d_ = self.cfg_denoise_func(self.denoiser3d, model_kwargs, scheduler_3d, enable_cfg)

            # Do the magic triangulation here, which will overwrite x0_3d_
            inputs["diffusion_step"] = (i, t3d, timesteps_3d)
            x0_3d_, x_triag, triag_info = self.triangulate_x0(x0_3d_, inputs, obs_c_p2d, obs_cr_p3d)
            # x0_3d_, x_triag, triag_info = self.triangulate_x0(x0_3d_, inputs)  # DEBUG: pure generation

            # Update: compute the previous noisy sample x_{t-1} and the original sample x_0
            x3d = scheduler_3d.step(x0_3d_, t3d, x3d, **extra_step_kwargs_3d).prev_sample

            # Store intermediate results
            if i % self.record_interval == 0:
                pred_progress.append(x_triag)  # (B, L, 22, 3)
                prior_progress.append(triag_info["denoiser_prior"])  # (B, 1, L, 22, 3)

            # progress bar
            prog_bar.update()

        # Post-processing
        outputs["triag_info"] = triag_info
        outputs["T_ayfz2c"] = self.T_ayfz2c
        _, pred_ayfz_motion, _ = self.triangulate_x0(x3d, inputs)
        self.T_ayfz2c = None

        # pred_ayfz_motion = self.decoder_motion3d(x3d)  # (B, L, 22, 3)
        outputs["pred_ayfz_motion"] = pred_ayfz_motion  # (B, L, 22, 3)
        outputs["pred_ayfz_motion_progress"] = torch.stack(pred_progress, dim=1)  # (B, Progress, L, 22, 3)
        outputs["pred_ayfz_motion_prior_progress"] = torch.stack(prior_progress, dim=1)  # (B, Progress, L, 22, 3)
        return outputs

    def triangulate_x0(self, x0_3d_, inputs, obs_c_p2d=None, obs_cr_p3d=None):
        """
        Args:
            x0_3d_: tensor that will be handled by self.decode and self.encode, shape example (B, D, L)
            c_p2d_obs : (B, L, 22, 2)
            obs_cr_p3d : (B, L, 22, 3)
        """
        info = {}  # record intermediate outputs

        # motion 3d prior
        device = x0_3d_.device
        pred_ayfz_motion = self.decoder_motion3d(x0_3d_)  # (B, L, 22, 3)
        info["denoiser_prior"] = pred_ayfz_motion.clone()[:, None]  # denoiser output
        B, L, J, _ = pred_ayfz_motion.shape

        ds_i, ds_t, ds_timesteps = inputs["diffusion_step"]
        ds_percent = ds_i / ds_timesteps[0]
        percent_start = 0.5
        if ds_percent < percent_start or (obs_c_p2d is None and obs_cr_p3d is None):
            # x0_3d_ = self.encoder_motion3d(pred_ayfz_motion.clone(), length=inputs["length"])  # (B, D, L)
            return x0_3d_, pred_ayfz_motion, info

        # wis3d= make_wis3d(name=inputs["meta"][0][0])
        wis3d = None

        # observation: 2d, 3d(cr or ay, this is dependent on the input 3d initialization)
        if obs_c_p2d is not None:
            # Compute T_ayfz2c
            if self.T_ayfz2c is None:
                # DEBUG: use gt_T_ayfz2c
                # self.T_ayfz2c = inputs["gt_T_ayfz2c"][:, None]  # (B, 1, 4, 4)

                # Use first 3 frames to initialization our method
                if "T_c1torest" not in inputs:
                    N_frames = 3
                    Ks = np.eye(3)[None].repeat(B, axis=0)  # (B, 3, 3)
                    kp2d = obs_c_p2d[:, :N_frames].cpu().numpy().reshape(B, -1, 2)
                    kp3d = pred_ayfz_motion[:, :N_frames].cpu().numpy().reshape(B, -1, 3)
                    fit_R, fit_t = ransac_PnP_batch(Ks, kp2d, kp3d, err_thr=0.2)
                    T_ayfz2c = transform_mat(torch.FloatTensor(fit_R), torch.FloatTensor(fit_t))[:, None]
                    self.T_ayfz2c = T_ayfz2c.to(device)  # (B, 1, 4, 4)
                else:
                    N_frames = 1
                    Ks = np.eye(3)[None].repeat(B, axis=0)  # (B, 3, 3)
                    kp2d = obs_c_p2d[:, :N_frames].cpu().numpy().reshape(B, -1, 2)
                    kp3d = pred_ayfz_motion[:, :N_frames].cpu().numpy().reshape(B, -1, 3)
                    fit_R, fit_t = ransac_PnP_batch(Ks, kp2d, kp3d, err_thr=0.2)
                    T_ayfz2c = transform_mat(torch.FloatTensor(fit_R), torch.FloatTensor(fit_t))[:, None]
                    self.T_ayfz2c = inputs["T_c1torest"] @ T_ayfz2c.to(device)  # (B, F, 4, 4)

                # Ks = np.eye(3)[None].repeat(B, axis=0)  # (B, 3, 3)
                # kp2d = obs_c_p2d[:, 0].cpu().numpy()
                # kp3d = obs_cr_p3d[:, 0].cpu().numpy()
                # fit_R, fit_t = ransac_PnP_batch(Ks, kp2d, kp3d, err_thr=0.2)
                # T_cr2c = transform_mat(torch.FloatTensor(fit_R), torch.FloatTensor(fit_t)).to(device)  # (B, 4, 4)
                # T_cr2c = T_cr2c[:, None]
                # # (B,1, 1, 1), (B,1, 3, 3), (B,1, 3, 1)
                # (scale, R_cr2ayfz), t_cr2ayfz = similarity_transform_batch(obs_cr_p3d[:, :1], pred_ayfz_motion[:, :1])
                # # self.scale_of_obs_cr_p3d = scale
                # T_cr2ayfz = transform_mat(R_cr2ayfz, t_cr2ayfz)
                # self.T_ayfz2c = T_cr2c @ torch.inverse(T_cr2ayfz)
                # self.T_ayfz2c = T_ayfz2c.to(pred_ayfz_motion.device)[:, None]

            T_ayfz2c = self.T_ayfz2c  # (B, 1/F, 4, 4)

            # Add cam-p2d-ray to info
            AddCamP2dRay = False
            if AddCamP2dRay:
                jids = [0, 7, 8, 15, 20, 21]
                info["w_cam_p2d_ray"] = compute_cam_p2d_ray(T_ayfz2c[:, 0], obs_c_p2d, jids)  # (B, L, 2, J, 3)
                info["T_ayfz2c"] = T_ayfz2c[:, 0]
                info["obs_c_p2d"] = obs_c_p2d
                info["length"] = inputs["length"]

            # === Global alignment === #
            GlobalAlignment = True
            if GlobalAlignment:
                length = inputs["length"]

                # max_iter = int(50 * (1 - ds_percent) + 5 * ds_percent) # linear from 50 to 5, by ds_percent 0.5 to 1
                # max_iter = 50
                # w, s, t = find_wst_transform(pred_ayfz_motion, T_ayfz2c, obs_c_p2d, length, obs_cr_p3d=obs_cr_p3d)
                w, s, t, t_init = find_wst_transform(pred_ayfz_motion, T_ayfz2c, obs_c_p2d, length, wis3d=wis3d)

                # w = self.make_smooth(w, dim=1, length=length)  # not work!
                # t = self.make_smooth(t, dim=1, length=length)
                motion_guidance = update_pred_with_wst(pred_ayfz_motion, w, s, t)

        # if obs_cr_p3d is not None:
        #     obs_cr_p3d = obs_cr_p3d - obs_cr_p3d[:, :, :1]
        #     (scale, R), t = similarity_transform_batch(obs_cr_p3d, motion_guidance)
        #     # smooth
        #     scale = self.make_smooth(scale, dim=1, length=length)
        #     R = rotation_6d_to_matrix(self.make_smooth(matrix_to_rotation_6d(R), dim=1, length=length))
        #     t = self.make_smooth(t, dim=1, length=length)
        #     # use (sR * cr_motion + t) as guidance
        #     motion_guidance_2 = einsum(obs_cr_p3d, scale * R, "b l j c, b l d c -> b l j d") + t[..., None, :, 0]
        #     motion_guidance = (motion_guidance_2 + motion_guidance) / 2

        # add_motion_as_lines(pred_ayfz_motion[0], wis3d, name="prior")
        # add_motion_as_lines(motion_guidance[0], wis3d, name="motion_guidance_2dlifted")
        # cam_oc = self.T_ayfz2c.inverse()[..., :3, 3]  # (B, L, 3)
        # cam_dir = self.T_ayfz2c.inverse()[..., :3, [2, 3]].sum(-1)  # (B, L, 3)
        # for f in range(L):
        #     wis3d.set_scene_id(f)
        #     wis3d.add_lines(cam_oc[0, [f]], cam_dir[0, [f]], name="cam", colors=np.array([[200, 0, 0]]))
        #     wis3d.add_point_cloud(cam_oc[0, [f]], colors=np.array([[255, 0, 0]]), name="cam_oc")
        #     wis3d.add_point_cloud(t_init[0, [f]], colors=np.array([[0, 0, 255]]), name="t_init")

        # obs_c_p2d_z1 = F.pad(obs_c_p2d, (0, 1), 'constant', 1)
        # obs_w_p2d_z1 = apply_T_on_points(obs_c_p2d_z1, T_ayfz2c.inverse())
        # add_motion_as_lines(obs_w_p2d_z1[0], wis3d, name='obs_c_p2d_z1')

        if obs_cr_p3d is not None:
            T_c2ayfz = T_ayfz2c.inverse()
            obs_pseudo_ayfz_p3d = apply_T_on_points(obs_cr_p3d, T_c2ayfz)
            obs_ayfz_p3d = obs_pseudo_ayfz_p3d - obs_pseudo_ayfz_p3d[:, :, :1] + motion_guidance[:, :, :1]
            # obs_ayfz_p3d = self.make_smooth(obs_ayfz_p3d, dim=1, length=length)
            motion_guidance = (obs_ayfz_p3d + motion_guidance) / 2

        # add_motion_as_lines(obs_ayfz_p3d[0], wis3d, name="motion_guidance_3d")

        # 2d jittering only uses 3d
        # jitter_mask = inputs["jitter_mask"]  # (B, L)
        # pred_ayf_motion_[jitter_mask] = pred_ayf_motion_prior[jitter_mask] # directly merging 3d also causes jitter
        # linear interpolation, a dirty code but work a little
        # for i in range(B):
        #     for j in range(1, L - 1):
        #         prev_m = pred_ayf_motion_[i, j - 1]
        #         j_ = j + 1
        #         while j_ < L:
        #             if not jitter_mask[i, j_]:
        #                 break
        #             j_ += 1
        #         next_m = pred_ayf_motion_[i, j_]
        #         if jitter_mask[i, j]:
        #             pred_ayf_motion_[i, j] = ((j_ - j) * prev_m + 1 * next_m) / (j_ - j + 1)

        # Add weights
        # x_triag = pred_ayfz_motion + (motion_guidance - pred_ayfz_motion) * (1 - ds_percent)
        x_triag = motion_guidance
        x0 = self.encoder_motion3d(x_triag.clone(), length=inputs["length"])  # (B, D, L)
        # x0[:, :67] = x0_3d_[:, :67] + (x0[:, :67] - x0_3d_[:, :67]) * (1 - ds_percent)
        x0[:, 67:] = x0_3d_[:, 67:]  # Do not change velocity terms, this may make motion very jittery

        return x0, x_triag, info

    def make_smooth(self, x, dim=-1, length=None):
        """x (..., f, ...) f at dim"""
        rad = self.kernel_smooth.size(-1) // 2
        x = x.clone()
        if length is not None:  # Assume B is the first dim
            x = x.transpose(dim, 1)
            for b in range(len(length)):
                x[b, length[b] : length[b] + rad] = x[b, length[b] - 1]
            x = x.transpose(1, dim)

        x = x.transpose(dim, -1)
        x_shape = x.shape[:-1]
        x = rearrange(x, "... f -> (...) 1 f")
        x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
        x = F.conv1d(x, self.kernel_smooth)
        x = x.squeeze(1).reshape(*x_shape, -1)
        x = x.transpose(-1, dim)
        return x


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


# def triangulate_x0_3d_only(self, x0_3d_, inputs):
#     JOINT_ID = [0, 1, 2, 3]

#     pred_ayf_motion = self.decoder_motion3d(x0_3d_)  # (B, L, 22, 3)
#     B, L, J, _ = pred_ayf_motion.shape
#     pred_ayf_motion_bkp = rearrange(pred_ayf_motion, "b l j c -> b 1 l j c").clone()

#     # Use init motion as 3D observation
#     init_motion_ayfz = inputs["init_motion_ayfz"]  # (B, L, 22, 3)
#     # Let's do local alignment before triangulation
#     # R, t = kabsch_algorithm_batch(init_motion_ayfz[:, :, JOINT_ID], pred_ayf_motion[:, :, JOINT_ID])
#     R, t = similarity_transform_batch(init_motion_ayfz[:, :, JOINT_ID], pred_ayf_motion[:, :, JOINT_ID])
#     init_motion_ayfz = apply_T_on_points(init_motion_ayfz, transform_mat(R, t))

#     # def update_ema(ema, new_value, alpha):
#     #     return alpha * new_value + (1 - alpha) * ema

#     # # # Smooth R, t by EMA
#     # span = 30
#     # alpha = 2 / (span + 1)  # span为你选择的时间窗口大小
#     # if self.ema == None:
#     #     self.ema = init_motion_ayfz
#     # self.ema = update_ema(self.ema, init_motion_ayfz, alpha)
#     # init_motion_ayfz = self.ema

#     # # fmt: off
#     # diff_ori = (pred_ayf_motion - inputs["init_motion_ayfz"])[:, :, JOINT_ID].pow(2).sum(-1).sqrt().sum(-1)
#     # diff_new = (pred_ayf_motion - init_motion_ayfz)[:, :, JOINT_ID].pow(2).sum(-1).sqrt().sum(-1)
#     # # fmt: on

#     pred_ayf_motion_ = (pred_ayf_motion + init_motion_ayfz) / 2  # equal to triangulation
#     # pred_ayf_motion_ = pred_ayf_motion  # equal to sampling
#     x_triag = pred_ayf_motion_.clone()
#     x0_inpaint = self.encoder_motion3d(pred_ayf_motion_, length=inputs["length"])
#     return x0_inpaint, x_triag, pred_ayf_motion_bkp


def compute_cam_p2d_ray(T_w2c, c_p2d, joints_to_record):
    """(B, 4, 4), (B, L, J, 2), list"""
    c_p2d_select = c_p2d[:, :, joints_to_record]
    B, L, J, _ = c_p2d_select.shape
    c_p2d_select = rearrange(c_p2d_select, "b l j c -> b (l j) c")

    # line_point + line_dir * x = point
    # c-coordinate
    c_ld = F.pad(c_p2d_select, (0, 1), value=1.0)  # (B, N, 3)
    c_lp = torch.zeros_like(c_ld)

    # w-coordinate
    assert len(T_w2c.shape) == 3
    T_c2w = torch.inverse(T_w2c)  # (B, 4, 4)
    R_c2w = T_c2w[:, :3, :3]  # (B, 3, 3)
    t_c2w = T_c2w[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    w_lp = einsum(R_c2w, c_lp, "b c d, b n d -> b n c") + t_c2w  # (B, N, 3)
    w_ld = einsum(R_c2w, c_ld, "b c d, b n d -> b n c")

    # solve y=0
    w_lend_y0 = torch.zeros_like(c_p2d_select[:, :, 1])
    x = (w_lend_y0 - w_lp[:, :, 1]) / w_ld[:, :, 1]  # (B, N)
    w_lend_y0 = w_lp + x.unsqueeze(-1) * w_ld  # (B, N, 3)

    w_lp = rearrange(w_lp, "b (l j) c -> b l j c", l=L, j=J)
    w_lend_y0 = rearrange(w_lend_y0, "b (l j) c -> b l j c", l=L, j=J)
    w_cam_p2d_ray = torch.stack([w_lp, w_lend_y0], dim=2)  # (B, L, 2, J, 3)

    return w_cam_p2d_ray
