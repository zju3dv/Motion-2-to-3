import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
from einops import einsum, rearrange, repeat

from hmr4d.utils.o3d_utils import o3d_skeleton_animation


# For rich test with 3d prior
class MotionObsCameraVisualizer(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            l = length[b]
            p_motion_prog = outputs["pred_ayfz_motion_progress"][b][:, :l]  # (progress, L, J, 3)
            txt = text[b] if text is not None else ""
            P = p_motion_prog.shape[0]

            w2c = []
            pos_2d = []
            if "gt_T_ayfz2c" in batch.keys():
                gt_w2c = batch["gt_T_ayfz2c"][b]  # (1, 4, 4)
                w2c.append(gt_w2c)
            elif "gt_T_w2c" in batch.keys():
                gt_w2c = batch["gt_T_w2c"][b, :l]  # (F, 4, 4)
                # w2c.append(gt_w2c)
            if "gt_c_p2d" in batch.keys():
                gt_pos_2d = batch["gt_c_p2d"][b, :l][None].expand(P, -1, -1, -1)  # (progress, L, J, 2)
                pos_2d.append(gt_pos_2d)
            else:
                pred_pos_2d = batch["obs_c_p2d"][b, :l][None].expand(P, -1, -1, -1)  # (progress, L, J, 2)
                # pos_2d.append(pred_pos_2d)

            # visualize used w2c
            p_w2c = outputs["triag_info"]["T_ayfz2c"][b, :l]  # (1/F, 4, 4)
            w2c.append(p_w2c)
            # visualize used c_p2d
            pred_pos_2d = batch["obs_c_p2d"][b, :l][None].expand(P, -1, -1, -1)  # (progress, L, J, 2)
            pos_2d.append(pred_pos_2d)

            pred_pos = []
            # vis used 3d obs
            obs_3d_pos = (
                batch["init_motion_ayfz"][b, :l][None].expand(P, -1, -1, -1)
                if "init_motion_ayfz" in batch.keys()
                else None
            )  # (progress, L, J, 3)
            if obs_3d_pos is None:
                obs_3d_pos = (
                    batch["pred_w_motion3d"][b, :l][None].expand(P, -1, -1, -1)
                    if "pred_w_motion3d" in batch.keys()
                    else None
                )  # (progress, L, J, 3)
                obs_3d_pos = obs_3d_pos[..., :22, :]
            if obs_3d_pos is not None:
                pred_pos.append(obs_3d_pos)

            obs_pseudo_3d_pos = (
                outputs["triag_info"]["obs_pseudo_ayfz_p3d"][b, :l][None].expand(P, -1, -1, -1)
                if "obs_pseudo_ayfz_p3d" in outputs["triag_info"].keys()
                else None
            )  # (progress, L, J, 3)

            if obs_pseudo_3d_pos is not None:
                pred_pos.append(obs_pseudo_3d_pos)
            prior_pos = outputs["pred_ayfz_motion_prior_progress"][b][:, :l]  # (progress, L, J, 3)
            pred_pos.append(prior_pos)
            pred_pos = torch.stack(pred_pos, dim=1)

            if "gt_ayfz_motion" in batch.keys():
                gt_pos = batch["gt_ayfz_motion"][b, :l][None, None].expand(P, -1, -1, -1, -1)  # (progress, 1, L, J, 3)
            elif "gt_w_motion3d" in batch.keys():
                gt_pos = batch["gt_w_motion3d"][b, :l]  # (L, J, 3)
                # EMDB gt has 24 joints
                gt_pos = gt_pos[..., :22, :]
                gt_pos = gt_pos - gt_pos[:1, :1]
                gt_pos = gt_pos[None, None].expand(P, -1, -1, -1, -1)  # (progress, 1, L, J, 3)
            else:
                gt_pos = None

            w2c = torch.stack(w2c, dim=1)  # (1/F, 2, 4, 4)

            pos_2d = torch.stack(pos_2d, dim=1)  # (progress, 2, L, J, 2)

            obs_cr_p3d = batch["obs_cr_p3d"][b, :l][None]  # (1, L, J, 3)
            c_pos = torch.stack([obs_cr_p3d] * pos_2d.shape[1], dim=1)  # (1, 2, L, J, 3)
            c_pos = c_pos.expand(P, -1, -1, -1, -1)  # (progress, 2, L, J, 3)

            if "cr_triag_progress" in outputs.keys():
                cr_p3d = outputs["cr_triag_progress"][b, :, :l]  # (progress, L, J, 3)
                c_pos = torch.stack([cr_p3d] * pos_2d.shape[1], dim=1)  # (progress, 2, L, J, 3)

            o3d_skeleton_animation(
                p_motion_prog,
                pos_2d=pos_2d,
                w2c=w2c,
                pred_pos=pred_pos,
                gt_pos=gt_pos,
                c_pos=c_pos,
                is_pinhole=True,
                name=txt,
            )
            self.cur_batch += 1


# For 2d 3d assisted generation
class MotionCameraVisualizer(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            l = length[b]
            p_motion_prog = outputs["pred_ayfz_motion_progress"][b][:, :l]  # (progress, L, J, 3)
            txt = text[b] if text is not None else ""
            P = p_motion_prog.shape[0]

            p_pos_2d = outputs["pred_motion_2d_progress"][b][:, :, :l]  # (progress, V, L, J, 2)
            w2c = outputs["pred_w2c_progress"][b][:, :l]  # (progress, L, V, 4, 4)

            gt_pos = batch["gt_ayfz_motion"][b, :l][None, None].expand(P, -1, -1, -1, -1)  # (progress, 1, L, J, 3)

            o3d_skeleton_animation(
                p_motion_prog,
                pos_2d=p_pos_2d,
                w2c=w2c,
                gt_pos=gt_pos,
                is_pinhole=False,
                name=txt,
            )
            self.cur_batch += 1
