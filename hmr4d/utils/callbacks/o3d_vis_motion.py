import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
from einops import einsum, rearrange, repeat

from hmr4d.utils.o3d_utils import o3d_skeleton_animation


class MotionVisualizer(pl.Callback):
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
            p_motion_prog = outputs["pred_smpl_progress"][b][:, :l]  # (progress, L, J, 3)

            p_motion_from_localjoints_prog = (
                outputs["pred_localjoints_progress"][b][:, None, :l]
                if "pred_localjoints_progress" in outputs.keys()
                else None
            )  # (progress, 1, L, J, 3)
            P = p_motion_prog.shape[0]
            # hmlvec has gt in batch
            gt_pos = (
                batch["gt_ayfz_motion"][b, :l][None, None].expand(P, -1, -1, -1, -1)
                if "gt_ayfz_motion" in batch.keys()
                else None
            )  # (progress, 1, L, J, 3)
            # smplpose has gt in outputs
            gt_pos = (
                outputs["gt_ayfz_motion"][b, :l][None, None].expand(P, -1, -1, -1, -1)
                if "gt_ayfz_motion" in outputs.keys() and gt_pos is None
                else None
            )  # (progress, 1, L, J, 3)
            txt = text[b] if text is not None else ""
            o3d_skeleton_animation(
                p_motion_prog,
                pred_pos=p_motion_from_localjoints_prog,
                gt_pos=gt_pos,
                name=txt,
            )
            self.cur_batch += 1
