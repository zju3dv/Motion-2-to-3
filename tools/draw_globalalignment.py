import torch
import numpy as np
from pprint import pprint

from hmr4d.utils.hml3d import (
    convert_motion_to_hmlvec263,
    convert_hmlvec263_to_motion,
    REPEAT_LAST_FRAME,
    ZERO_FRAME_AHEAD,
    readable_hml263_vec,
)
from hmr4d.utils.o3d_utils import o3d_skeleton_animation
from hmr4d.model.supermotion.utils.motion3d_endecoder import Hmlvec263EnDecoder
from hmr4d.utils.pylogger import Log
from hmr4d.utils.skeleton_motion_visualization import SkeletonAnimationGenerator
import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface, cartesian_to_spherical


def get_3dbbox_corners(pos):
    min_bound = pos.reshape(-1, 3).min(dim=0)[0]
    max_bound = pos.reshape(-1, 3).max(dim=0)[0]
    # compute eight corners of the AABB given min_bound and max_bound
    x_corners = torch.stack(
        [
            min_bound[0],
            max_bound[0],
            max_bound[0],
            min_bound[0],
            min_bound[0],
            max_bound[0],
            max_bound[0],
            min_bound[0],
        ]
    )
    y_corners = torch.stack(
        [
            min_bound[1],
            min_bound[1],
            max_bound[1],
            max_bound[1],
            min_bound[1],
            min_bound[1],
            max_bound[1],
            max_bound[1],
        ]
    )
    z_corners = torch.stack(
        [
            min_bound[2],
            min_bound[2],
            min_bound[2],
            min_bound[2],
            max_bound[2],
            max_bound[2],
            max_bound[2],
            max_bound[2],
        ]
    )
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)
    return corners


def get_2dbbox_corners(pos, scale=1.0):
    min_bound = pos.reshape(-1, 3).min(dim=0)[0]
    max_bound = pos.reshape(-1, 3).max(dim=0)[0]
    # compute four corners of the AABB given min_bound and max_bound
    x_corners = torch.stack(
        [
            min_bound[0],
            max_bound[0],
            max_bound[0],
            min_bound[0],
        ]
    )
    y_corners = torch.stack(
        [
            min_bound[1],
            min_bound[1],
            max_bound[1],
            max_bound[1],
        ]
    )
    z_corners = torch.zeros_like(x_corners)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)
    center = corners.mean(dim=0, keepdim=True)
    corners = (corners - center) * scale + center
    return corners


if __name__ == "__main__":
    motion = torch.load("./x.pth")  # L, J, 3
    motion = torch.from_numpy(motion).float()[:1]
    distance = torch.ones((1,)) * 5
    angle = torch.ones((1,)) * -0.3 * torch.pi
    ele_angle = torch.ones((1,)) * 0.15 * torch.pi
    # ele_angle = None
    cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, ele_angle)  # 1, 4, 4
    T_w2c = torch.inverse(cam_mat)  # 1, 4, 4
    c_motion = matrix.get_relative_position_to(motion, cam_mat)  # F, J, 3
    c_motion = c_motion[..., :3] / (c_motion[..., 2:] + 1e-4)  # perspective
    w_c_motion = matrix.get_position_from(c_motion, cam_mat)  # F, J, 3 for save
    c_motion = c_motion[..., :2]  # remove z

    bbox_2d = get_2dbbox_corners(motion, scale=1.5)  # 4, 3
    bbox_2d = matrix.get_position_from_rotmat(bbox_2d, cam_mat[:, :3, :3])
    cam_bbox_2d = matrix.get_relative_position_to(bbox_2d, cam_mat)  # 4, 3
    cam_bbox_2d = cam_bbox_2d[..., :3] / (cam_bbox_2d[..., 2:] + 1e-4)  # perspective
    w_cam_bbox_2d = matrix.get_position_from(cam_bbox_2d, cam_mat)  # 4, 3 for save
    cam_bbox_2d = cam_bbox_2d[..., :2]  # remove z

    prior_angle = torch.rand((1,)) * 2 * torch.pi
    y_axis = torch.zeros(prior_angle.shape[:-1] + (3,), device=prior_angle.device)
    y_axis[..., 1] = 1.0
    prior_rot = matrix.quat_from_angle_axis(prior_angle, y_axis)  # [*, 4]
    prior_rotmat = matrix.rot_matrix_from_quaternion(prior_rot)  # [*, 3, 3]

    prior_pos = motion.clone()
    prior_pos = matrix.get_position_from_rotmat(prior_pos, prior_rotmat)
    prior_pos = prior_pos * 0.9
    prior_pos[..., 0] += torch.ones((1,)) * 1
    prior_pos[..., 2] += torch.ones((1,)) * 1

    bbox = get_3dbbox_corners(motion)
    prior_bbox = matrix.get_position_from_rotmat(bbox, prior_rotmat[0])
    prior_bbox = prior_bbox * 0.9
    prior_bbox[..., 0] += torch.ones((1,)) * 1
    prior_bbox[..., 2] += torch.ones((1,)) * 1
    bbox_3d_edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    bbox_2d_edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ]

    data = {
        "target_pos": motion.numpy(),
        "prior_pos": prior_pos.numpy(),
        "target_bbox": bbox.numpy(),
        "prior_bbox": prior_bbox.numpy(),
        "camera_pos": w_c_motion.numpy(),
        "camera_bbox": w_cam_bbox_2d.numpy(),
        "w2c": T_w2c.numpy(),
        "3d_edges": bbox_3d_edges,
        "2d_edges": bbox_2d_edges,
    }
    np.save("./globalalignment.npy", data)

    o3d_skeleton_animation(motion, pos_2d=c_motion[None], gt_pos=prior_pos, w2c=T_w2c, is_pinhole=True)
