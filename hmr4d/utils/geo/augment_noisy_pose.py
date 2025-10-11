import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
import hmr4d.utils.matrix as matrix


def gaussian_augment(body_pose, std_angle=10.0, to_R=True):
    """
    Args:
        body_pose torch.Tensor: (..., J, 3) axis-angle if to_R is True, else rotmat (..., J, 3, 3)
        std_angle: scalar or list, in degree
    """

    body_pose = body_pose.clone()

    if to_R:
        body_pose_R = axis_angle_to_matrix(body_pose)  # (B, L, J, 3, 3)
    else:
        body_pose_R = body_pose
    shape = body_pose_R.shape[:-2]
    device = body_pose.device

    # 1. Simulate noise
    # angle:
    std_angle = torch.tensor(std_angle).to(device).reshape(-1)  # allow scalar or list
    noise_angle = torch.randn(shape, device=device) * std_angle * torch.pi / 180

    # axis: avoid zero vector
    noise_axis = torch.rand((*shape, 3), device=device)
    mask_ = torch.norm(noise_axis, dim=-1) < 1e-6
    noise_axis[mask_] = 1

    noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
    noise_aa = noise_angle[..., None] * noise_axis  # (B, L, J, 3)
    noise_R = axis_angle_to_matrix(noise_aa)  # (B, L, J, 3, 3)

    # 2. Add noise to body pose
    new_body_pose_R = matrix.get_mat_BfromA(body_pose_R, noise_R)  # (B, L, J, 3, 3)
    # new_body_pose_R = torch.matmul(noise_R, body_pose_R)
    new_body_pose_r6d = matrix_to_rotation_6d(new_body_pose_R)  # (B, L, J, 6)
    new_body_pose_aa = matrix_to_axis_angle(new_body_pose_R)  # (B, L, J, 3)

    return new_body_pose_R, new_body_pose_r6d, new_body_pose_aa
