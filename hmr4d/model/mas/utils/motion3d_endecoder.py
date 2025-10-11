import torch
import numpy as np
from hmr4d.utils.hml3d import (
    convert_motion_to_hmlvec263,
    convert_hmlvec263_to_motion,
    convert_motion_to_hmlvec263_original,
)

from .motion2d_endecoder import EnDecoderBase
from einops import rearrange, einsum, repeat


class Hmlvec263OriginalEnDecoder(EnDecoderBase):
    def __init__(self, stats_module, stats_name):
        """Original implementation"""
        super().__init__(stats_module, stats_name)
        self.mean = self.mean.reshape(263)
        self.std = self.std.reshape(263)

    def encode(self, data, length):
        """
        Args:
            data: (B, L, J, 3), in ayfz coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, 263, L)
        """
        data = data.clone()
        B, L, _, _ = data.shape
        all_hmlvec263 = torch.zeros((B, L, 263), device=data.device)
        for i in range(B):
            l = length[i]
            # We pad the last frame to get the same length, slightly different from the original
            data_i = torch.cat([data[i, :l], data[i, l - 1 : l]], dim=0)
            hmlvec263, _, _, _ = convert_motion_to_hmlvec263_original(data_i.detach().cpu().numpy())
            all_hmlvec263[i, :l] = torch.from_numpy(hmlvec263).to(data.device)
        x = (all_hmlvec263 - self.mean) / self.std

        return x.permute(0, 2, 1)
