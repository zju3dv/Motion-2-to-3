import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, get_const_colors
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.geo_transform import T_transforms_points, project_p2d
from einops import rearrange, repeat, einsum
from hmr4d.dataset.supermotion.collate import pad_to_max_len


class MotionLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.mid_logged = {}  # a motion id to wis3d
        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["B"]
        meta = batch["meta"]

        length = batch["length"]
        # gt_ayfz_motion = batch["gt_ayfz_motion"]  # (B, Lmax, 22, 3)
        pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, Lmax, 22, 3)
        obs_cr_p3ds = batch["pred_cr_motion3d"].cpu()  # (B, Lmax, 22, 3)

        # info = {k: v.cpu() for k, v in outputs["triag_info"].items()}
        # out = {**info, "meta": meta, "gt_ayfz_motion": gt_ayfz_motion.cpu()}
        # torch.save(out, "info.pt")

        for b in range(B):
            mid = meta[b][0].replace("/", "-")
            if mid not in self.mid_logged:
                self.mid_logged[mid] = make_wis3d(name=mid)
            wis3d = self.mid_logged[mid]

            start, end = meta[b][1]
            l = end - start
            pred = pred_ayfz_motion[b, :l]
            # gt = gt_ayfz_motion[b, :l]
            obs_cr_p3d = obs_cr_p3ds[b, :l] + torch.tensor([3.0, 0, 0])

            # add_motion_as_lines(obs_cr_p3d, wis3d, name="obs_cr_p3d", const_color="blue", offset=start)
            # add_motion_as_lines(gt, wis3d, name="gt_ayfz_motion", const_color="green", offset=start)
            # add_motion_as_lines(pred, wis3d, name="prior3d", offset=start)
            # add_motion_as_lines(pred, wis3d, name="pred_gp_root_first", offset=start)

            add_motion_as_lines(pred, wis3d, name="pred", offset=start)
            # add_motion_as_lines(pred, wis3d, name="prior3d_new", offset=start)
            # add_motion_as_lines(pred, wis3d, name="pred + obs_cr_p3d", offset=start)

            # se_lines = outputs["triag_info"]["w_cam_p2d_ray"][b, :l]
            # color = get_const_colors("orange", (se_lines.size(-2),))[:, :3] * 255
            # for f in range(l):
            #     wis3d.set_scene_id(f + start)
            #     wis3d.add_lines(se_lines[f, 0], se_lines[f, 1], color, name="pred-cam-p2d-ray")
