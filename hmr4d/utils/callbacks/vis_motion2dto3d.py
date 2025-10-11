import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
import hmr4d.utils.matrix as matrix
from hmr4d.dataset.supermotion.collate import pad_to_max_len
from einops import einsum, rearrange, repeat
import numpy as np

from hmr4d.utils.o3d_utils import o3d_skeleton_animation, pos_2dto3d, get_good_z_for_2dvis


class Motion2Dto3DVisualizer(pl.Callback):
    def __init__(self, skeleton_type="smpl"):
        """Visualizing final motion."""
        super().__init__()
        self.skeleton_type = skeleton_type
        self.cur_batch = 0

        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)
        pred_motion_prog = outputs["pred_progress"]["pred_motion"]  # (B, progress, L, J, 3)
        pred_motion2d_prog = outputs["pred_progress"]["pred_motion2d"]  # (B, progress, V, L, J, 2)
        # gt_motion = batch["gt_motion"]  # (B, L, J, 3)
        is_pinhole = batch["is_pinhole"][0]
        T_w2c = batch["T_w2c"]  # (B, V, 4, 4)
        J = pred_motion2d_prog.shape[-2]
        P = pred_motion2d_prog.shape[1]

        for b in range(B):
            l = length[b]
            # gt_motion_ = gt_motion[b][:l]  # L, J, 3
            pred_m_prog = pred_motion_prog[b][:, :l]  # progress, L, J, 3
            pred_m2d_prog = pred_motion2d_prog[b][:, :, :l]  # progress, V, L, J, 2
            T_w2c_ = T_w2c[b]
            txt = text[b] if text is not None else ""

            # FIXME: hardcode joints number here
            #if False:
            if J == 23:
                # Assume last joints is next frame root
                accum_root = torch.cumsum(pred_m_prog[..., -1:, :], dim=-3)  # (progress, L, 1, 3)
                # Convert accumlated world pose to true world pose
                accum_pred_m_prog = pred_m_prog.clone()
                # Add accumulated root to each joints instead of virtual next frame root
                accum_pred_m_prog[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
                # Assign virtual next frame root
                accum_pred_m_prog[..., -1:, :] = accum_root

                cam_mat = torch.inverse(T_w2c_)  # V, 4, 4
                cam_mat = repeat(cam_mat, "v c d -> p l v c d", p=P, l=l)  # (progress, l, V, 4, 4)
                cam_pos = matrix.get_position(cam_mat)  # progress, l, V, 3
                accum_cam_pos = cam_pos + accum_root  # (progress, l, V, 3)
                accum_cam_mat = matrix.set_position(cam_mat, accum_cam_pos)  # (progress, l, V, 4, 4)
                T_w2c_ = torch.inverse(accum_cam_mat)  # (progress, l, V, 4, 4)

                o3d_skeleton_animation(
                    accum_pred_m_prog,
                    pos_2d=pred_m2d_prog,
                    w2c=T_w2c_,
                    # gt_pos=gt_motion_,
                    is_pinhole=is_pinhole,
                    name="Global-" + txt,
                    skeleton_type=self.skeleton_type,
                )
            else:
                o3d_skeleton_animation(
                    pred_m_prog,
                    pos_2d=pred_m2d_prog,
                    w2c=T_w2c_,
                    # gt_pos=gt_motion_,
                    is_pinhole=is_pinhole,
                    name="Local-" + txt,
                    skeleton_type=self.skeleton_type,
                )

            self.cur_batch += 1


class Wis3DMotion2Dto3DVisualizer(pl.Callback):
    def __init__(self, name, max_len=200, max_count=100, plot_2D=False, subset=None):
        super().__init__()
        self.name = name
        self.max_len = max_len
        self.max_count = max_count  # wis3d costs a lot cpu
        self.plot_2D = plot_2D
        self.subset = subset
        self.on_test_batch_end = self.on_predict_batch_end
        self.counter = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)
        pred_motion_prog = outputs["pred_progress"]["pred_motion"]  # (B, progress, L, J, 3)
        pred_motion2d_prog = outputs["pred_progress"]["pred_motion2d"]  # (B, progress, V, L, J, 2)
        if "pred_single2d" in outputs["pred_progress"]:
            pred_single2d_prog = outputs["pred_progress"]["pred_single2d"]  # (B, progress, L, J, 2)
        else:
            pred_single2d_prog = None
        # is_pinhole = batch["is_pinhole"][0]
        is_pinhole = trainer.model.pipeline.args.is_perspective
        T_w2c = batch["T_w2c"]  # (B, V, 4, 4)
        if "gt_motion" in batch:
            gt_motion = batch["gt_motion"]  # (B, L, J, 3)
        else:
            gt_motion = None

        pred_motion = outputs["pred_global_motion"]  # (B, L, 22, 3)
        B, L, J, _ = pred_motion.shape
        # ay to ayfz
        T_ay2ayfz = compute_T_ayfz2ay(pred_motion[:, 0], inverse=True)  # (B, 4, 4)
        pred_motion_ = rearrange(pred_motion, "b l j c -> b (l j) c")  # (B, L*J, 3)
        pred_ayfz_motion = apply_T_on_points(pred_motion_, T_ay2ayfz)  # (B, L*22, 3)
        pred_ayfz_motion = rearrange(pred_ayfz_motion, "b (l j) c -> b l j c", j=J)  # (B, L, 22, 3)
        if gt_motion is not None:
            gt_L = gt_motion.shape[1]
        else:
            gt_L = 300
        pred_ayfz_motion_ = torch.zeros((B, gt_L, J, 3), device=pred_ayfz_motion.device)
        for i, l in enumerate(length):
            pred_ayfz_motion_[i, :l] = pred_ayfz_motion[i, :l]  # pad with last frame
            pred_ayfz_motion_[i, l:] = pred_ayfz_motion[i, l - 1]  # pad with last frame
        pred_ayfz_motion_floor = pred_ayfz_motion.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
        pred_ayfz_motion_[..., 1] = pred_ayfz_motion_[..., 1] - pred_ayfz_motion_floor[:, None, None]

        J = pred_motion2d_prog.shape[-2]
        P = pred_motion2d_prog.shape[1]

        for b in range(B):
            if self.counter > self.max_count:
                continue

            l = length[b]
            if gt_motion is not None:
                gt_motion_ = gt_motion[b][:l]  # L, J, 3
            pred_m_prog = pred_motion_prog[b][:, :l]  # progress, L, J, 3
            pred_m2d_prog = pred_motion2d_prog[b][:, :, :l]  # progress, V, L, J, 2
            if pred_single2d_prog is not None:
                pred_s2d_prog = pred_single2d_prog[b][:, :l]  # progress, L, J, 2
            T_w2c_ = T_w2c[b]
            txt = text[b] if text is not None else ""
            print(f"\n{self.counter:03d}_" + txt)

            # FIXME: hardcode joints number here
            if J == 23:
                # Assume last joints is next frame root
                accum_root = torch.cumsum(pred_m_prog[..., -1:, :], dim=-3)  # (progress, L, 1, 3)
                # Convert accumlated world pose to true world pose
                accum_pred_m_prog = pred_m_prog.clone()
                # Add accumulated root to each joints instead of virtual next frame root
                accum_pred_m_prog[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
                # Assign virtual next frame root
                accum_pred_m_prog[..., -1:, :] = accum_root

                cam_mat = torch.inverse(T_w2c_)  # V, 4, 4
                cam_mat = repeat(cam_mat, "v c d -> p l v c d", p=P, l=l)  # (progress, l, V, 4, 4)
                cam_pos = matrix.get_position(cam_mat)  # progress, l, V, 3
                accum_cam_pos = cam_pos + accum_root  # (progress, l, V, 3)
                accum_cam_mat = matrix.set_position(cam_mat, accum_cam_pos)  # (progress, l, V, 4, 4)
                T_w2c_ = torch.inverse(accum_cam_mat)  # (progress, l, V, 4, 4)

                # remove next root joint
                pred_m = accum_pred_m_prog[-1, :, :-1]  # (L, 22, 3)
                pred_m2d = pred_m2d_prog[-1, :, :, :-1]  # (V, L, 22, 2)
                if pred_single2d_prog is not None:
                    pred_s2d = pred_s2d_prog[-1, :, :-1] # (L, 22, 2)
            else:
                pred_m = pred_m_prog[-1]  # (L, J, 3)
                pred_m2d = pred_m2d_prog[-1]  # V, L, J, 2
                if pred_single2d_prog is not None:
                    pred_s2d = pred_s2d_prog[-1] # (L, J, 2)

            T_w2c_last = rearrange(T_w2c_[-1], "l v c d -> v l c d")  # V, L, 4, 4
            root = pred_m[None, :, 0, :]  # 1, L, 3
            _, min_z = get_good_z_for_2dvis(T_w2c_last, root, is_pinhole=is_pinhole)
            pred_m2din3d = pos_2dto3d(pred_m2d, T_w2c_last, min_z, is_pinhole=is_pinhole)  # V, L, J, 3

            if pred_single2d_prog is not None:
                pred_s2din3d = pos_2dto3d(pred_s2d, T_w2c_last[0], min_z[0], is_pinhole=is_pinhole)  # L, J, 3

            mid = txt
            mid = mid.replace("/", "_")
            mid = mid[:200]
            pred = pad_to_max_len(pred_m, self.max_len)
            if self.subset is not None:
                wis3d = make_wis3d(name=f"{self.counter:03d}_" + mid[:20], output_dir=f"outputs/wis3d_{self.subset}")
            else:
                wis3d = make_wis3d(name=f"{self.counter:03d}_" + mid[:20])
            if gt_motion is not None:
                gt = pad_to_max_len(gt_motion_, self.max_len)
                add_motion_as_lines(gt, wis3d, name="00gt ::: " + mid, const_color="green")
            if self.name == "ours":
                add_motion_as_lines(pred_ayfz_motion_[b], wis3d, name=f"01{self.name}_pred ::: " + mid, const_color="blue")
            else:
                # use "mas" causes very strange bug
                add_motion_as_lines(pred_ayfz_motion_[b], wis3d, name=f"01{self.name}_pred ::: " + mid, const_color="magenta")
            if self.plot_2D:
                add_motion_as_lines(pred, wis3d, name="02pred ::: " + mid)
                for i in range(pred_m2din3d.shape[0]):
                    pred_m2din3d_v = pad_to_max_len(pred_m2din3d[i], self.max_len)
                    add_motion_as_lines(pred_m2din3d_v, wis3d, name=f"03view-{i} ::: {mid}", const_color="red")
                
                if pred_single2d_prog is not None:
                    pred_s2din3d = pad_to_max_len(pred_s2din3d, self.max_len)
                    add_motion_as_lines(pred_s2din3d, wis3d, name=f"04single ::: {mid}", const_color="cyan")

            self.counter += 1
