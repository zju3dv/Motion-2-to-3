import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log

from diffusers.schedulers import DDPMScheduler
from hmr4d.utils.diffusion.pipeline_helper import PipelineHelper
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase, SMPLEnDecoder, SMPLRelVecV51EnDecoder
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam, compute_transl_full_cam, project_to_bi01
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo.optim_smplify import run_smplify
from hmr4d.model.smplify.losses import SimpleSMPLifyLoss
from hmr4d.utils.smplx_utils import make_smplx


class PipelineMinimalMocap(nn.Module, PipelineHelper):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.num_inference_steps = args.num_inference_steps
        self.enable_record_progress = args.enable_record_progress
        if self.enable_record_progress:
            assert args.num_visualize <= self.num_inference_steps
            self.record_interval = args.num_inference_steps // args.num_visualize

        # ----- Scheduler ----- #
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = instantiate(args.scheduler_opt_sample)
        self.guidance_scale = args.guidance_scale

        # ----- Networks ----- #
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        Log.info(self.denoiser3d)

        self.wo_diffusion = args.get("wo_diffusion", False)
        if self.wo_diffusion:
            Log.info("!!!!! Without DIFFUSION: use learned initialization instead of noise !!!!!")
            max_len = self.denoiser3d.max_len
            d_model = self.denoiser3d.input_dim
            # learnable embedding (C, L)
            self.noisy_x = nn.Parameter(torch.randn(d_model, max_len), requires_grad=True)
            assert self.num_inference_steps == 1

        # Functions for En/Decoding from motion to x, including normalization
        self.data_endecoder: SMPLRelVecV51EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        self.encoder_mainvec = self.data_endecoder.encode  # main vector for motion3d
        self.decoder_mainvec = self.data_endecoder.decode
        # extra utilities

        # ----- Loss weights ----- #
        self.weights = args.weights

        # ----- Guidance ----- #
        self.loss_fn = SimpleSMPLifyLoss(make_smplx("supermotion"))

    # ========== Training ========== #
    @staticmethod
    def build_model_kwargs(x, timesteps, length, f_condition, enable_cfg=False):
        if enable_cfg:
            length = torch.cat([length, length])
            for k in f_condition.keys():
                f_condition[k] = torch.cat([torch.zeros_like(f_condition[k]), f_condition[k]])
        return dict(x=x, timesteps=timesteps, length=length, **f_condition)

    def forward_train(self, inputs):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        global_orient_incam = inputs["global_orient_incam"]  # (B, L, 3)
        scheduler = self.tr_scheduler
        B, L, _ = global_orient_incam.shape

        # *. set beta, skeleton...
        self.data_endecoder.set_cfg(inputs)

        # *. get ayfz inputs for global motion...
        ayfz_inputs = self.data_endecoder.convert2ayfzdata(inputs)

        # *. Encoding
        x = self.encoder_mainvec(ayfz_inputs)  # (B, C, L)

        # *. Get noisy observation
        obs_r6d = self.data_endecoder.get_noisyobs(ayfz_inputs)  # (B, L, J, 6)
        f_noisyobs = self.data_endecoder.normalize_local_pose_r6d(obs_r6d)

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)
        if self.wo_diffusion:
            noisy_x = repeat(self.noisy_x, "C L -> B C L", B=B)

        # Conditions
        assert self.training
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_condition = {
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_noisyobs": f_noisyobs,  # (B, L, C)
            "f_cam_angvel": inputs["cam_angvel"],  # (B, L, C=6)
        }
        f_condition = randomly_set_null_condition(f_condition)

        # Forward
        model_kwargs = self.build_model_kwargs(x=noisy_x, timesteps=t, length=length, f_condition=f_condition)
        model_output = self.denoiser3d(**model_kwargs)
        model_pred = model_output.sample  # (B, C, L)
        mask = model_output.mask  # (B, 1, L)
        pred_cam = model_output.extra_output.permute(0, 2, 1)  # (B, L, 3)
        transl_incam = compute_transl_full_cam(pred_cam, inputs["bbx_xys"], inputs["K_fullimg"])

        # ========== Compute Loss ========== #
        total_loss = 0

        # 1. Simple loss: MSE
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

        # 2. Extra loss
        decode_dict = self.decoder_mainvec(model_pred)

        # Incam loss
        decode_dict.update(
            {
                "transl_incam": transl_incam,  # (B, L, 3)
                "mask": mask,
            }
        )
        extra_incam_loss, extra_incam_loss_dict = compute_extra_incam_loss(
            inputs, decode_dict, self.data_endecoder, self.weights
        )
        total_loss += extra_incam_loss
        outputs.update(extra_incam_loss_dict)

        # Global loss
        extra_global_loss, extra_global_loss_dict = compute_extra_global_loss(
            ayfz_inputs, decode_dict, self.data_endecoder, self.weights
        )
        total_loss += extra_global_loss
        outputs.update(extra_global_loss_dict)

        outputs["loss"] = total_loss
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

        x0_ = noise_pred
        return x0_, denoiser_out

    # ========== Sample ========== #
    def forward_sample(self, inputs, capture_mode=False):
        # Setup
        outputs = dict()
        generator = inputs["generator"]
        assert self.guidance_scale == 1, "We only support guidance_scale=1 for now"
        enable_cfg = False if self.guidance_scale == 1 else True
        scheduler = self.te_scheduler
        self.data_endecoder.clear_ayfz()

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

        max_L = self.denoiser3d.max_len
        obs_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(inputs["obs"].reshape(B, -1, 21, 3)))  #  Last-padding
        f_noisyobs = self.data_endecoder.normalize_local_pose_r6d(obs_r6d)  # (B, L, C)

        # 1. Prepare target variable x, which will be denoised progressively
        x = self.prepare_x(shape=(B, self.denoiser3d.input_dim, max_L), generator=generator)
        if self.wo_diffusion:
            x = repeat(self.noisy_x, "C L -> B C L", B=B)

        # 2. Conditions
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_condition = {
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_noisyobs": f_noisyobs,  # (B, L, C)
            "f_cam_angvel": inputs["cam_angvel"],  # (B, L, C=6)
        }

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler, generator)  # for scheduler.step()
        pred_progress = {}  # for visualization
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x, timesteps=t, length=length, f_condition=f_condition, enable_cfg=enable_cfg
            )
            x0_, denoiser_out = self.cfg_denoise_func(self.denoiser3d, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            if t < self.args.guidance_after:  # default 0
                x = self.guide_x(x, denoiser_out, inputs)

            if self.enable_record_progress and i % self.record_interval == 0:
                decode_dict = self.decoder_mainvec(x0_)
                if "smpl" not in pred_progress.keys():
                    pred_progress["smpl"] = []
                    pred_progress["localjoints"] = []
                    pred_progress["incam_localjoints"] = []
                ayfz_smpl = self.data_endecoder.fk_forward(**decode_dict)[0]
                ayfz_localjoints = self.data_endecoder.localjoints_forward(**decode_dict)[0]

                pred_cam = denoiser_out.extra_output.permute(0, 2, 1)  # (B, L, 3)
                transl_c = compute_transl_full_cam(pred_cam, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
                incam_localjoints = self.data_endecoder.localjoints_forward(
                    transl_c,
                    decode_dict["global_orient_incam"],
                    decode_dict["local_joints"],
                )[0]

                pred_progress["smpl"].append(ayfz_smpl)
                pred_progress["localjoints"].append(ayfz_localjoints)
                pred_progress["incam_localjoints"].append(incam_localjoints)

            # progress bar
            prog_bar.update()

        # Post-processing
        decode_dict = self.decoder_mainvec(x)

        # Predicted Cam
        pred_cam = denoiser_out.extra_output.permute(0, 2, 1)  # (B, L, 3)
        transl_c = compute_transl_full_cam(pred_cam, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)

        if self.enable_record_progress:
            for k in pred_progress.keys():
                outputs["pred_" + k + "_progress"] = torch.stack(pred_progress[k], dim=1)  # (B, Progress, L, J, 3)

        # -> pred_cr_j3d
        outputs["pred_smpl_params_incam"] = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "global_orient": decode_dict["global_orient_incam"],  # (B, L, 3)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "transl": transl_c,  # (B, L, 3)
        }
        outputs["pred_smpl_params_global"] = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "global_orient": decode_dict["global_orient"],  # (B, L, 3)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "transl": decode_dict["transl"],  # (B, L, 3)
        }
        outputs["pred_cam"] = pred_cam
        s_transl_vel, ayfz_global_orient = self.data_endecoder.decode_raw_global_root(x, decode_dict)
        outputs["pred_s_transl_vel"] = s_transl_vel
        outputs["pred_ayfz_global_orient"] = ayfz_global_orient

        return outputs

    def guide_x(self, x0_, denoiser_out, inputs):
        info = {}  # record intermediate outputs
        B, C, L = x0_.shape

        # Naive impl a smplify on params_incam (global_orient, body_pose, betas, transl), and then partially overwrite x0_
        decode_dict = self.data_endecoder.decode(x0_)

        if False:  # Check local_joints difference
            wis3d = make_wis3d(name="debug-guidance")
            local_joints_fk = self.data_endecoder.fk_v2(decode_dict["body_pose"], decode_dict["betas"])
            local_joints_fk = local_joints_fk - local_joints_fk[:, :, :1]  # (B, L, J, 3)
            local_joints_net = self.data_endecoder.denorm(x0_[:, :212])[..., 9:72].reshape(B, L, 21, 3)
            local_joints_net = torch.cat([local_joints_fk[..., :1, :], local_joints_net], dim=-2)  # Add a pseudo root
            add_motion_as_lines(local_joints_fk[0], wis3d, name="local-joints-fk")
            add_motion_as_lines(local_joints_net[0], wis3d, name="local-joints-net")

            local_joint_diff = (local_joints_fk[:, :, 1:] - local_joints_net[:, :, 1:]).abs().mean()
            # after normalization
            mean = self.data_endecoder.mean[9:72]
            std = self.data_endecoder.std[9:72]
            fk_ = (local_joints_fk[:, :, 1:].flatten(2) - mean) / std
            net_ = (local_joints_net[:, :, 1:].flatten(2) - mean) / std

        # A temp-fix
        denorm_x0_ = self.data_endecoder.denorm(x0_[:, :212])
        ayfz_transl_vel, ayfz_global_orient_rot6d = denorm_x0_[..., :3], denorm_x0_[..., 3:9]

        if False:  # Find max error dim when doing a reverse check
            x0_recover = self.data_endecoder.encode_a_decode(decode_dict, ayfz_transl_vel, ayfz_global_orient_rot6d)
            error = (x0_recover[0, :, 0] - x0_[0, :, 0]).abs()  # (C, )
            max_val, max_dim = error.max(0)
            Log.info(f"max_val={max_val}, max_dim={max_dim}")

        # Overwrite
        pred_cam = denoiser_out.extra_output.permute(0, 2, 1)  # (B, L, 3)
        transl_c = compute_transl_full_cam(pred_cam, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        param_dict = {
            "body_pose": decode_dict["body_pose"],
            "global_orient": decode_dict["global_orient_incam"],
            "betas": decode_dict["betas"],
            "transl": transl_c,
        }
        helper_dict = {
            "bbx_xys": inputs["bbx_xys"],
            "K_fullimg": inputs["K_fullimg"],
            "kp2d": inputs["kp2d"],
        }
        smplify_out = run_smplify(param_dict, helper_dict, self.loss_fn)
        decode_dict["betas"] = smplify_out["betas"]
        decode_dict["body_pose"] = smplify_out["body_pose"]
        decode_dict["global_orient_incam"] = smplify_out["global_orient"]

        x0_ = self.data_endecoder.encode_a_decode(decode_dict, ayfz_transl_vel, ayfz_global_orient_rot6d)

        return x0_


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """
    To support classifier-free guidance, randomly set-to-unconditioned,
    Conditions are in shape (B, L, C)
    """
    B = f_condition["f_noisyobs"].size(0)

    keys = list(f_condition.keys())
    for k in keys:
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(B, f_condition[k].shape[1]) < uncond_prob
        f_condition[k][mask] = 0.0

    return f_condition


def compute_extra_incam_loss(inputs, decode_dict, data_endecoder, weights):
    extra_loss_dict = {}
    extra_loss = 0
    mask = decode_dict["mask"]  # effective length mask

    # Incam FK
    # prediction
    fk_dict = {
        "body_pose": decode_dict["body_pose"],  # (B, L, 63)
        "betas": decode_dict["betas"],  # (B, L, 10)
        "global_orient": decode_dict["global_orient_incam"],  # (B, L, 3)
        "transl": decode_dict["transl_incam"],  # (B, L, 3)
    }
    pred_c_j3d = data_endecoder.fk_v2(**fk_dict)
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)

    # gt
    L = inputs["body_pose"].size(1)
    fk_dict = {
        "body_pose": inputs["body_pose"],  # (B, L, 63)
        "betas": inputs["betas"][:, None].expand(-1, L, -1),  # (B, L, 10)
        "global_orient": inputs["global_orient_incam"],  # (B, L, 3)
        "transl": inputs["transl_incam"],  # (B, L, 3)
    }
    gt_c_j3d = data_endecoder.fk_v2(**fk_dict)  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    if weights.cr_j3d > 0.0:
        if mask is not None:
            pred_cr_j3d = pred_cr_j3d * mask[:, 0, :, None, None]
            gt_cr_j3d = gt_cr_j3d * mask[:, 0, :, None, None]
        cr_j3d_loss = F.mse_loss(pred_cr_j3d.float(), gt_cr_j3d.float(), reduction="mean")
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    if weights.transl_c > 0.0:
        pred_transl = decode_dict["transl_incam"]  # (B, L, 3)
        gt_transl = inputs["transl_incam"]
        if mask is not None:
            pred_transl = pred_transl * mask[:, 0, :, None]
            gt_transl = gt_transl * mask[:, 0, :, None]
        # transl_c_loss = F.mse_loss(pred_transl.float(), gt_transl.float(), reduction="mean")
        transl_c_loss = F.l1_loss(pred_transl.float(), gt_transl.float(), reduction="mean")
        extra_loss += transl_c_loss * weights.transl_c
        extra_loss_dict["transl_c_loss"] = transl_c_loss

    if weights.j2d > 0.0:
        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, J, 2)
        if mask is not None:
            pred_j2d_01 = pred_j2d_01 * mask[:, 0, :, None, None]
            gt_j2d_01 = gt_j2d_01 * mask[:, 0, :, None, None]
        j2d_loss = F.mse_loss(pred_j2d_01.float(), gt_j2d_01.float(), reduction="mean")

        if False:
            if j2d_loss > 20:
                if pred_j2d_01.max() > 100 or pred_j2d_01.min() < -100:
                    B, L, J, C = pred_j2d_01.shape
                    max_val, flat_index = pred_j2d_01.reshape(-1).max(0)
                    B_index = flat_index // (L * J * C)
                    Log.warn(f"pred_j2d_01: max={max_val}, Bid= {B_index}")
                    Log.warn(f"data-meta: {inputs['meta'][B_index]}")

                if gt_j2d_01.max() > 1.1:
                    B, L, J, C = gt_j2d_01.shape
                    max_val, flat_index = gt_j2d_01.reshape(-1).max(0)
                    B_index = flat_index // (L * J * C)
                    Log.warn(f"gt_j2d_01: max={max_val}, Bid= {B_index}")
                    Log.warn(f"data-meta: {inputs['meta'][B_index]}")

                print("StopHere")

        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        raise NotImplementedError
        # SMPL forward
        pred_cr_verts, pred_cr_j3d = self.data_endecoder.get_pred_cr_verts(model_pred)
        gt_cr_smpl_params = {
            "body_pose": inputs["smplpose"][:, :, 6:],  # (B, L, 63)
            "global_orient": inputs["global_orient_incam"],  # (B, L, 3)
            "betas": beta[:, None].expand(-1, L, -1),  # (B, L, 10)
        }
        gt_cr_verts, gt_cr_j3d = self.data_endecoder.get_gt_cr_verts(gt_cr_smpl_params)
        if mask is not None:  # length mask
            pred_cr_verts = pred_cr_verts * mask.reshape(B, L, 1, 1)
            pred_cr_j3d = pred_cr_j3d * mask.reshape(B, L, 1, 1)
            gt_cr_verts = gt_cr_verts * mask.reshape(B, L, 1, 1)
            gt_cr_j3d = gt_cr_j3d * mask.reshape(B, L, 1, 1)
        cr_vert_loss = F.mse_loss(pred_cr_verts.float(), gt_cr_verts.float(), reduction="mean")

    if False:  # visualize
        wis3d = make_wis3d(name="debug-extraloss")
        bid = 0

        # gt_skeleton = self.data_endecoder.smplx_model.get_skeleton(inputs["beta"][:, None].expand(-1, L, -1))
        # add_motion_as_lines(gt_skeleton[bid, :1], wis3d, name="gt-skeleton")
        add_motion_as_lines(gt_c_j3d[bid], wis3d, name="gt-c-j3d")

        # SMPLX forward to check
        smplx_model = self.data_endecoder.smplx_model
        smplx_out = smplx_model(**{k: v[bid] for k, v in fk_dict.items()})
        gt_c_j3d_smplx = smplx_out.joints[:, :22]
        add_motion_as_lines(gt_c_j3d_smplx, wis3d, name="gt-c-j3d-smplx")

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, decode_dict, data_endecoder, weights):
    extra_loss_dict = {}
    extra_loss = 0
    mask = decode_dict["mask"]  # effective length mask

    if weights.fk_pos > 0.0 or weights.fk_vel > 0.0 or weights.fk_contact > 0.0:
        pos_mask = mask[:, 0, :, None, None]  # (B, L, 1, 1)
        vel_mask = pos_mask[:, 1:]  # (B, L-1, 1, 1)
        contact_mask = vel_mask  # (B, L-1, 1, 1)

        gt_ayfz_data = data_endecoder.get_ayfz_data()

        gt_transl = gt_ayfz_data["transl"]
        gt_body_pose = gt_ayfz_data["body_pose"]
        gt_joints_pos = gt_ayfz_data["joints"]
        gt_joints_vel = gt_joints_pos[:, 1:] - gt_joints_pos[:, :-1]  # (B, L-1, J, 3)
        contact_label = data_endecoder.detect_foot_contact(gt_joints_pos)[:, :-1]  # (B, L-1, 4)
        contact_label = contact_label[..., None]  # (B, L-1, 4, 1)  # TODO: ~static labels -> every joint

        gt_joints_pos = gt_joints_pos * pos_mask
        gt_joints_vel = gt_joints_vel * vel_mask
        contact_label = contact_label * contact_mask

        ########## Global Forward kinematics loss #############
        fk_dict = {
            "transl": gt_transl,  # (B, L, 3)
            "global_orient": decode_dict["global_orient"],  # (B, L, 3)
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "betas": decode_dict["betas"],  # (B, L, 10)
        }

        pred_fk_pos = data_endecoder.forward_func(**fk_dict)[0]  # (B, L, J, 3)
        pred_fk_pos = pred_fk_pos * pos_mask
        pred_fk_vel = pred_fk_pos[:, 1:] - pred_fk_pos[:, :-1]  # (B, L-1, J, 3)
        pred_fk_vel = pred_fk_vel * vel_mask
        pred_fkfoot_vel = data_endecoder.get_foot_vel(pred_fk_vel)  # (B, L-1, 4, 3)
        pred_fkfoot_vel_square = pred_fkfoot_vel * pred_fkfoot_vel * contact_mask

        fkpos_loss = F.mse_loss(pred_fk_pos.float(), gt_joints_pos.float(), reduction="mean")
        fkvel_loss = F.mse_loss(pred_fk_vel.float(), gt_joints_vel.float(), reduction="mean")
        fkcontact_loss = (contact_label * pred_fkfoot_vel_square).mean()
        extra_loss_dict["fkpos_loss"] = fkpos_loss
        extra_loss += fkpos_loss * weights.fk_pos
        extra_loss_dict["fkvel_loss"] = fkvel_loss
        extra_loss += fkvel_loss * weights.fk_vel
        extra_loss_dict["fkcontact_loss"] = fkcontact_loss
        extra_loss += fkcontact_loss * weights.fk_contact

    return extra_loss, extra_loss_dict
