import torch
import torch.nn as nn
import importlib
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, compute_root_quaternion_ay
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from einops import rearrange, einsum
from hmr4d.utils.smplx_utils import make_smplx
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.augment_noisy_pose import gaussian_augment
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


# fmt: off
class EnDecoderBase(nn.Module):
    def __init__(self, stats_module, stats_name) -> None:
        super().__init__()
        try:
            stats = getattr(importlib.import_module(stats_module), stats_name)
            Log.info(f"We use {stats_name} for statistics!")
            self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("std", torch.tensor(stats["std"]).float(), False)
        except Exception as e:
            print(e)
            Log.info(f"Cannot find {stats_name} in {stats_module}! Use zero as mean, one as std!")
            self.register_buffer("mean", torch.zeros(1).float(), False)
            self.register_buffer("std", torch.ones(1).float(), False)

    def encode(self, motion, length=None): raise NotImplemented  # (B, L, J, 3) -> (B, L, D)  TODO: Need discussion
    def decode(self, x): raise NotImplemented   # (B, L, D) -> (B, L, J, 3)
# fmt: on


class EuclideanEnDecoder(EnDecoderBase):
    def __init__(self, stats_module, stats_name, J=22, C=2):
        """Simple gaussian standardization"""
        super().__init__(stats_module, stats_name)
        self.J = J
        self.C = C
        self.mean = self.mean.reshape(-1)
        self.std = self.std.reshape(-1)

    def encode(self, data):
        """
        Args:
            data: (B, L, J, C)
        Returns:
            x: (B, J*C, L)
        """
        x = data.flatten(-2)  # (B, L, J, C) -> (B, L, J*C)
        # TODO: set default number
        x = self.norm(x)  # (B, J*C, L)
        return x

    def decode(self, x):
        """Reverse process of encode"""
        x = self.denorm(x)  # (B, L, J*C)
        data = x.reshape(x.shape[:-1] + (self.J, self.C))  # (B, L, J*C) -> (B, L, J, C)
        return data

    def norm(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            x: (B, C, L)
        """
        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.transpose(-1, -2)  # (B, C, L)
        return x

    def denorm(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            x: (B, L, C)
        """
        x = x.transpose(-1, -2)  # (B, L, C)
        x = x * self.std + self.mean  # (B, L, C)
        return x
