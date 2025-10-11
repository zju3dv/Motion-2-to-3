import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, add_prog_motion_as_lines
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.geo_transform import T_transforms_points, project_p2d
from einops import rearrange, repeat, einsum


class ProgMotionLogger(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=32):
        """Visualizing final motion."""
        super().__init__()
        self.wis3d = make_wis3d(name=name, time_postfix=time_postfix)
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        L = check_equal_get_one(batch["length"], "length")  # scalar
        pred_motion = outputs["pred_ayfz_motion_progress"]  # (B, P, L, J, 3)

        # Downsample progress
        P = pred_motion.shape[1]
        P_target = 10
        if P > P_target:
            p_index = torch.linspace(0, P - 1, P_target, dtype=torch.long)
            pred_motion = pred_motion[:, p_index, ...]

        # Downsample time: our motion is at 30fps, downsample to 7.5fps
        pred_motion = pred_motion[:, :, ::4, ...]

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")

            record_name = f"Id_{self.cur_batch:03d}"
            add_prog_motion_as_lines(pred_motion[b], self.wis3d, name=record_name, skeleton_type="smpl22")
            self.cur_batch += 1
