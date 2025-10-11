import torch
from pytorch3d.transforms import axis_angle_to_matrix


def R_from_wy(w, use_sincos=True):
    """Compute R from w, which is the rotation of y axis"""
    w = w * torch.pi
    if use_sincos:
        zeros = torch.zeros_like(w).detach()
        ones = torch.ones_like(w).detach()
        R = torch.stack([torch.cos(w), zeros, torch.sin(w), zeros, ones, zeros, -torch.sin(w), zeros, torch.cos(w)], -1)
        R = R.reshape(*R.shape[:-1], 3, 3)
    else:  # use axis_angle_to_matrix conversion
        R = axis_angle_to_matrix(torch.stack([torch.zeros_like(w), w, torch.zeros_like(w)], -1))
    return R
