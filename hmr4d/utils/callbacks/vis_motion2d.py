import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
from einops import einsum, rearrange, repeat

from hmr4d.utils.plt_utils import plt_skeleton_animation


class Motion2DVisualizer(pl.Callback):
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
        pred_motion2d_prog = outputs["pred_progress"]["pred_motion2d"]  # (B, progress, L, J, 2)
        gt_motion2d = batch["gt_motion2d"]  # (B, L, J, 2)
        # add root at zero
        gt_motion2d = torch.cat([torch.zeros_like(gt_motion2d[..., :1, :]), gt_motion2d], dim=-2)

        for b in range(B):
            l = length[b]
            gt_motion2d_ = gt_motion2d[b][:l]  # (L, J, 2)
            pred_m2d_prog_ = pred_motion2d_prog[b][:, :l]  # (progress, L, J, 2)
            txt = text[b] if text is not None else ""
            # plt_skeleton_animation(gt_motion2d_, text="gt\n" + txt, skeleton_type=self.skeleton_type)
            plt_skeleton_animation(pred_m2d_prog_, text=txt, skeleton_type=self.skeleton_type)
            self.cur_batch += 1
