import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
from einops import einsum, rearrange, repeat

from hmr4d.utils.o3d_utils import o3d_skeleton_animation


class MotionLogger(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        device = batch["length"].device
        text = batch.get("text", None)
        P = outputs["pred_motion_progress"].shape[1]

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            l = length[b]
            p_motion_prog = outputs["pred_motion_progress"][b][:, :l]  # (progress, L, 263)
            p_motion_prog = rearrange(p_motion_prog, "p l c -> p c l")  # (progress, 263, L)
            joints_pos = convert_hmlvec263_to_motion(p_motion_prog, abs_3d=False)  # (progress, L, 22, 3)
            txt = text[b] if text is not None else ""
            o3d_skeleton_animation(joints_pos, name=txt)
            self.cur_batch += 1
