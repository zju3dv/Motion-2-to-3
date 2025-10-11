import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.geo_transform import T_transforms_points, project_p2d
from einops import rearrange, repeat, einsum
from hmr4d.dataset.supermotion.collate import pad_to_max_len


class GenHml3dMotionLogger(pl.Callback):
    def __init__(self, name, max_len=300):
        super().__init__()
        self.wis3d = make_wis3d(name=name)  # , time_postfix=True)
        self.max_len = max_len
        self.on_test_batch_end = self.on_predict_batch_end
        self.counter = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["B"]
        length = batch["length"]
        gt_ayfz_motion = outputs["gt_ayfz_motion"]  # (B, Lmax, 22, 3)
        pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, Lmax, 22, 3)
        text = batch["text"]

        for b in range(B):
            # mid = text[b]
            mid = f"motion_{self.counter}"
            l = length[b]
            pred = pad_to_max_len(pred_ayfz_motion[b, :l], self.max_len)
            gt = pad_to_max_len(gt_ayfz_motion[b, :l], self.max_len)

            add_motion_as_lines(gt, self.wis3d, name=mid + "_gt", skeleton_type="smpl22")
            add_motion_as_lines(pred, self.wis3d, name=mid, skeleton_type="smpl22")

            # add_motion_as_lines(pred, self.wis3d, name=mid+" [prior3d-only]", skeleton_type="smpl22")
            self.counter += 1
            if self.counter == 4:
                breakpoint()
                print("Warning, hitting self.counter")
