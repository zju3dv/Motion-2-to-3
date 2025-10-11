import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, get_gradient_colors, get_const_colors
from .motion_wis3d_logger import convert_motion_to_colored_lines


class MotionLogger(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.wis3d = make_wis3d(name=name, time_postfix=time_postfix)
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

        self.kinematic_chain = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ]

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"][0]
        device = batch["length"].device

        green_const = torch.tensor([0, 1.0, 0, 1.0])[None].to(device)
        red_gradients = get_gradient_colors("red", num_points=length, alpha=0.6).to(device)
        green_gradients = get_gradient_colors("green", num_points=length, alpha=0.6).to(device)
        blue_gradients = get_gradient_colors("blue", num_points=length, alpha=0.6).to(device)

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            record_name = f"sample_{self.cur_batch:03d}"

            # ============== PointCloud ============== #
            points = torch.tensor([], device=device).reshape(0, 3)
            colors = torch.tensor([], device=device).reshape(0, 4)

            pred_traj_stage_2 = outputs["pred_motion"][b, :, 0].clone()  # (L, 3)
            pred_traj_stage_2[:, 1] = 0  # project to y=0
            cur_points = torch.cat([points, pred_traj_stage_2], dim=0)
            cur_colors = torch.cat([colors, blue_gradients], dim=0)

            # ============== Stage-2 ============== #
            # Skeleton as colored lines
            pred_motion = outputs["pred_motion"][b]  # (L, 22, 3)
            s_points, e_points, m_colors = convert_motion_to_colored_lines(pred_motion, self.kinematic_chain)

            # ============== Add to wis3d ================= #
            for f in range(120):
                self.wis3d.set_scene_id(f)
                self.wis3d.add_point_cloud(cur_points, cur_colors, name=f"{record_name}_stage2")
                self.wis3d.add_lines(s_points[f], e_points[f], m_colors[f], name=f"{record_name}_stage2")

            self.cur_batch += 1
