from typing import Any, Dict
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange

from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points, compute_T_ayf2az
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.eval.sliding_windows import (
    get_window_startends,
    split_pad_batch,
    slide_merge,
    slide_merge_root_aa_ayfz,
)
from pytorch3d.transforms import axis_angle_to_matrix


class MotionPriorPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        args=None,
        ignored_weights_prefix=["pipeline.clip"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.seed = args.seed
        self.rng_type = args.rng_type  # [const, random, seed_plus_idx]
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

    def training_step(self, batch, batch_idx):
        # forward and compute loss
        B = self.trainer.train_dataloader.batch_size
        outputs = self.pipeline.forward_train(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

        return outputs

    def validation_step(self, batch, batch_idx):
        """Task-specific validation step. CAP or GEN."""
        task = check_equal_get_one(batch["task"], "task")

        # Add generator to batch
        if "generator" not in batch:
            batch["generator"] = self.get_generatror(batch_idx)

        # Forward
        if task == "CAP-Seq":
            assert (
                batch["B"] == 1
            ), "Only support batch size 1 for CAP-Seq, we parrallelize the inference within the batch."
            seq_length = batch["length"][0].item()
            gender = batch["gender"][0]
            T_w2ay = batch["T_w2ay"][0]
            max_L = self.pipeline.denoiser3d.max_len  # assume 120
            overlap = 20

            batch_window = create_sliding_window(batch, seq_length, max_L, overlap, self.pipeline)
            outputs_window = self.pipeline.forward_sample(batch_window, capture_mode=True)
            outputs = merge_sliding_window(batch_window, outputs_window)

            if False:  # Wis
                wis3d = make_wis3d(name="debug-rich-cap")
                smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()

                # Prediction
                # add_motion_as_lines(outputs_window["pred_ayfz_motion"][bid], wis3d, name="pred_ayfz_motion")

                smplx_out = smplx_model(**pred_smpl_params_global)
                for i in range(len(smplx_out.vertices)):
                    wis3d.set_scene_id(i)
                    wis3d.add_mesh(smplx_out.vertices[i], smplx_model.bm.faces, name=f"pred-smplx-global")

                # GT (w)
                smplx_models = {
                    "male": make_smplx("rich-smplx", gender="male").cuda(),
                    "female": make_smplx("rich-smplx", gender="female").cuda(),
                }
                gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
                gt_smplx_out = smplx_models[gender](**gt_smpl_params)

                # GT (ayfz)
                smplx_verts_ay = apply_T_on_points(gt_smplx_out.vertices, T_w2ay)
                smplx_joints_ay = apply_T_on_points(gt_smplx_out.joints, T_w2ay)
                T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:1], inverse=True)[0]  # (4, 4)
                smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (F, 22, 3)

                for i in range(len(smplx_verts_ayfz)):
                    wis3d.set_scene_id(i)
                    wis3d.add_mesh(smplx_verts_ayfz[i], smplx_models[gender].bm.faces, name=f"gt-smplx-ayfz")

                breakpoint()

            # o3d vis
            if False:
                prog_keys = [
                    "pred_smpl_progress",
                    "pred_localjoints_progress",
                    "pred_incam_localjoints_progress",
                ]
                for k in prog_keys:
                    if k in outputs_window:
                        seq_out = torch.cat(
                            [v[:, :l] for v, l in zip(outputs_window[k], length)], dim=1
                        )  # (B, P, L, J, 3) -> (P, L, J, 3) -> (P, CL, J, 3)
                        outputs[k] = seq_out[None]

        else:
            assert task == "GEN", f"Task `{task}' is not supported."
            outputs = self.pipeline.forward_sample(batch)

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg is None:
            return optimizer

        scheduler_cfg = self.scheduler_cfg
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)

        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #

    def get_generatror(self, batch_idx):
        """Fix the random seed for each batch at sampling stage."""
        generator = torch.Generator(self.device)
        if self.rng_type == "const":
            generator.manual_seed(self.seed)
        elif self.rng_type == "random":
            pass
        elif self.rng_type == "seed_plus_idx":
            generator.manual_seed(self.seed + batch_idx)
        else:
            raise ValueError(f"rng_type `{self.rng_type}' is not supported.")
        generator.manual_seed(self.seed)
        return generator

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    Log.debug(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

        super().on_save_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")

        if ckpt_type == "pl_2d3dprior":
            assert len(ckpt_path) == 2
            state_dict = {
                **torch.load(ckpt_path[0], "cpu")["state_dict"],
                **torch.load(ckpt_path[1], "cpu")["state_dict"],
            }
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            real_missing = []
            for ig_keys in self.ignored_weights_prefix:
                for k in missing:
                    if not k.startswith(ig_keys):
                        real_missing.append(k)

            if len(real_missing) > 0:
                Log.warn(f"Missing keys: {real_missing}")
            if len(unexpected) > 0:
                Log.warn(f"Unexpected keys: {unexpected}")

        else:
            assert ckpt_type == "pl"
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            real_missing = []
            for ig_keys in self.ignored_weights_prefix:
                for k in missing:
                    if not k.startswith(ig_keys):
                        real_missing.append(k)

            if len(real_missing) > 0:
                Log.warn(f"Missing keys: {real_missing}")
            if len(unexpected) > 0:
                Log.warn(f"Unexpected keys: {unexpected}")


def create_sliding_window(batch, seq_length, max_L, overlap, pipeline):
    # Create sliding window
    startends = get_window_startends(seq_length, max_L, overlap)  # List of tuple

    # Split and pad
    length = torch.tensor([e - s for s, e in startends]).to(batch["f_imgseq"][0].device)
    f_imgseq = split_pad_batch(batch["f_imgseq"][0], max_L, startends)  # (B', L, 1024)
    cam_angvel = split_pad_batch(batch["cam_angvel"][0], max_L, startends)  # (B', L, 6)
    obs = split_pad_batch(batch["obs_smpl_params"]["body_pose"][0], max_L, startends)  # (B', L, 63)
    bbx_xys = split_pad_batch(batch["bbx_xys"][0], max_L, startends)  # (B', L, 4)
    K_fullimg = split_pad_batch(batch["K_fullimg"][0], max_L, startends)  # (B', L, 3, 3)
    kp2d = split_pad_batch(batch["kp2d"][0], max_L, startends)  # (B', L, J, 2)

    if False:  # Simulate obs as in training
        gt_obs = split_pad_batch(batch["gt_smpl_params"]["body_pose"][0], max_L, startends)  # (B', L, 63)
        obs = pipeline.data_endecoder.get_noisyobs({"body_pose": gt_obs}, "aa")  # (B', L, J, 3)

    batch_window = {
        "generator": batch["generator"],
        "startends": startends,
        "length": length,
        "f_imgseq": f_imgseq,
        "cam_angvel": cam_angvel,
        "obs": obs,
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "kp2d": kp2d,
    }

    return batch_window


def merge_sliding_window(batch_window, outputs_window):
    outputs = dict()
    startends = batch_window["startends"]
    special_vtype_dict = {
        "body_pose": "aa_j3",
        "global_orient": "aa_j3",
    }

    # incam motion
    outputs["pred_smpl_params_incam"] = {}
    for k in outputs_window["pred_smpl_params_incam"]:
        vtype = special_vtype_dict.get(k, "vec")
        outputs["pred_smpl_params_incam"][k] = slide_merge(
            outputs_window["pred_smpl_params_incam"][k], startends, vtype
        )

    # global motion
    outputs["pred_smpl_params_global"] = {
        "body_pose": outputs["pred_smpl_params_incam"]["body_pose"],
        "betas": outputs["pred_smpl_params_incam"]["betas"],
    }

    # Blend transl_vel and root_R
    pred_s_transl_vel = slide_merge(outputs_window["pred_s_transl_vel"], startends, "vec")
    global_orient = slide_merge_root_aa_ayfz(outputs_window["pred_ayfz_global_orient"], startends)
    global_orient_R = axis_angle_to_matrix(global_orient)

    # Rollout to global transl
    vel_global_orient_R = torch.cat([global_orient_R[:1], global_orient_R[:-1]], dim=0)
    transl = torch.zeros_like(pred_s_transl_vel)  # (L, 3)
    transl[1:] = pred_s_transl_vel[:-1]  # (L, 3)
    transl = torch.einsum("lij,lj->li", vel_global_orient_R, transl)  # (L, 3)
    transl = torch.cumsum(transl, dim=0)  # (L, 3)

    outputs["pred_smpl_params_global"]["global_orient"] = global_orient
    outputs["pred_smpl_params_global"]["transl"] = transl

    if False:
        wis3d = make_wis3d(name="debug-sliding-window")
        smplx = make_smplx("supermotion").cuda()
        smplx_out = smplx(**outputs["pred_smpl_params_global"])
        for i in range(len(smplx_out.vertices)):
            wis3d.set_scene_id(i)
            wis3d.add_mesh(smplx_out.vertices[i], smplx.faces, name=f"pred-global-sw")

        # # By default, simply put windows together, without blending
        #  window = outputs_window["pred_smpl_params_global"]
        # "global_orient": slide_merge(window["global_orient"], startends, do_blend=False),
        # "transl": slide_merge(window["transl"], startends, do_blend=False),
        # smplx_out = smplx(**outputs["pred_smpl_params_global"])
        # for i in range(len(smplx_out.vertices)):
        #     wis3d.set_scene_id(i)
        #     wis3d.add_mesh(smplx_out.vertices[i], smplx.faces, name=f"pred-global-wo_blending")

    outputs["pred_cam"] = slide_merge(outputs_window["pred_cam"], startends, "vec")

    return outputs
