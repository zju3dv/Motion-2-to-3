import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.geo_transform import T_transforms_points, project_p2d
from einops import rearrange, repeat, einsum


class DebugSeqMotionLogger(pl.Callback):
    def __init__(self, wis3d_dir):
        super().__init__()
        self.wis3d_dir = wis3d_dir
        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["B"]
        meta = batch["meta"]
        assert check_equal_get_one([m[0] for m in meta]), "can handle 1 sequence"

        # gt_ayfz_motion = batch["gt_ayfz_motion"]  # (B, Lmax, 22, 3)
        # pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, Lmax, 22, 3)
        gt_ayfz_motion = []
        pred_ayfz_motion = []

        vid = meta[0][0].replace("/", "_")
        wis3d = make_wis3d(output_dir=self.wis3d_dir, name=vid)

        for b in range(B):
            # Add gt motion
            _, (start, end) = meta[b]
            gt_ayfz_motion.append(batch["gt_ayfz_motion"][b, : end - start])
            pred_ayfz_motion.append(outputs["pred_ayfz_motion"][b, : end - start])

        gt_ayfz_motion = torch.cat(gt_ayfz_motion, dim=0)  # (F, 22, 3)
        pred_ayfz_motion = torch.cat(pred_ayfz_motion, dim=0)

        # add_motion_as_lines(gt_ayfz_motion, wis3d, name="gt_ayfz_motion", skeleton_type="smpl22")
        add_motion_as_lines(pred_ayfz_motion, wis3d, name="pred_00p2", skeleton_type="smpl22")
        # add_motion_as_lines(pred_ayfz_motion, wis3d, name="pred_wo_triag", skeleton_type="smpl22")
