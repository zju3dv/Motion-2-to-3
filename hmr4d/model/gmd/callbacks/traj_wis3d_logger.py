import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, get_gradient_colors


class TrajLogger(pl.Callback):
    def __init__(self, name, max_batches=1000):
        super().__init__()
        self.wis3d = make_wis3d(name=name, time_postfix=True)
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"][0]
        device = batch["length"].device

        green_const = torch.tensor([0, 1.0, 0, 1.0])[None].to(device)
        red_gradients = get_gradient_colors("red", num_points=length, alpha=0.6).to(device)
        green_gradients = get_gradient_colors("green", num_points=length, alpha=0.6).to(device)

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            record_name = f"sample_{self.cur_batch:03d}"

            # We will put everything into one pointcloud
            points = torch.tensor([], device=device).reshape(0, 3)
            colors = torch.tensor([], device=device).reshape(0, 4)

            # Cond root keypoints
            cond_rkpt = batch["cond_rkpt"][b][batch["cond_rkpt_mask"][b].sum(-1) != 0]
            points = torch.cat([points, cond_rkpt], dim=0)
            colors = torch.cat([colors, green_const.repeat(len(cond_rkpt), 1)], dim=0)

            #  Cond root trajectory
            cond_traj = batch["cond_traj"][b][batch["cond_traj_mask"][b].sum(-1) != 0]
            points = torch.cat([points, cond_traj], dim=0)
            colors = torch.cat([colors, green_gradients], dim=0)

            # Prediction (red gradient)
            for i, pred_traj in enumerate(outputs["pred_traj_progress"][b]):  # for each intermediate step
                self.wis3d.set_scene_id(i)
                cur_points = torch.cat([points, pred_traj], dim=0)
                cur_colors = torch.cat([colors, red_gradients], dim=0)
                self.wis3d.add_point_cloud(cur_points, cur_colors, name=record_name)

            self.cur_batch += 1
