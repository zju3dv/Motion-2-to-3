import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, get_gradient_colors, get_const_colors


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

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            record_name = f"sample_{self.cur_batch:03d}"

            # ============== PointCloud ============== #
            # root-keypoint-condition-projection (on y=0), pointcloud
            # root-trajectory-prediction-projection (on y=0), pointcloud
            # We will put everything into one pointcloud
            points = torch.tensor([], device=device).reshape(0, 3)
            colors = torch.tensor([], device=device).reshape(0, 4)

            # Cond root keypoints
            cond_rkpt = batch["cond_rkpt"][b][batch["cond_rkpt_mask"][b].sum(-1) != 0]
            points = torch.cat([points, cond_rkpt], dim=0)
            colors = torch.cat([colors, green_const.repeat(len(cond_rkpt), 1)], dim=0)

            # Prediction (red gradient)
            pred_traj = outputs["pred_traj"][b]  # (L, 3)
            pred_traj[:, 1] = 0  # project to y=0
            cur_points = torch.cat([points, pred_traj], dim=0)
            cur_colors = torch.cat([colors, red_gradients], dim=0)

            # ============== Stage-2 ============== #
            # Skeleton as colored lines
            pred_motion = outputs["pred_motion"][b]  # (L, 22, 3)
            s_points, e_points, m_colors = convert_motion_to_colored_lines(pred_motion, self.kinematic_chain)

            # ============== Add to wis3d ================= #
            for f in range(120):
                self.wis3d.set_scene_id(f)
                self.wis3d.add_point_cloud(cur_points, cur_colors, name=f"{record_name}_stage1")
                self.wis3d.add_lines(s_points[f], e_points[f], m_colors[f], name=f"{record_name}_stage2")

            self.cur_batch += 1


def convert_motion_to_colored_lines(skeleton, kinamatic_chain):
    """
    Args:
        skeleton (tensor): (L, 22, 3)
    Returns:
        s_points: (L, ?, 3)
        e_points: (L, ?, 3)
        m_colors: (L, ?, 3)
    """
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    length = skeleton.shape[0]
    device = skeleton.device
    for chain, color_name in zip(kinamatic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(skeleton[:, chain[:-1]])
        e_points.append(skeleton[:, chain[1:]])
        color_ = get_const_colors(color_name, partial_shape=(length, num_line), alpha=1.0).to(device)  # (120, 4, 4)
        m_colors.append(color_[..., :3] * 255)  # (120, 4, 3)
    s_points = torch.cat(s_points, dim=1)
    e_points = torch.cat(e_points, dim=1)
    m_colors = torch.cat(m_colors, dim=1)
    return s_points, e_points, m_colors
