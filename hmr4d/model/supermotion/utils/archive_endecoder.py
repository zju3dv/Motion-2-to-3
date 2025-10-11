import torch
import numpy as np
from hmr4d.utils.hml3d import (
    convert_motion_to_hmlvec263,
    convert_hmlvec263_to_motion,
    convert_motion_to_hmlvec263_original,
)
from hmr4d.utils.hml3d import ZERO_FRAME_AHEAD, REPEAT_LAST_FRAME

from .motion3d_endecoder import EnDecoderBase
from einops import rearrange, einsum, repeat


class EuclideanEnDecoder(EnDecoderBase):
    def __init__(self, stats_module, stats_name, J=22, C=3, data_shape="J C", x_shape="(J C)"):
        """Simple gaussian standardization"""
        super().__init__(stats_module, stats_name)
        self.J = J
        self.C = C
        self.data_shape = data_shape
        self.x_shape = x_shape
        self.mean = self.mean.reshape(J, C)
        self.std = self.std.reshape(J, C)

    def encode(self, data):
        """
        Args:
            data: (B, L, J, C)
        Returns:
            x: (B, J*C, L)
        """
        mean_shape = self.mean.shape
        assert data.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        x = (data - self.mean) / self.std
        x = rearrange(x, f"... {self.data_shape} -> ... {self.x_shape}")  # e.g. (B, L, J, C) -> (B, L, J*C)
        return x.permute(0, 2, 1)

    def decode(self, x):
        """Reverse process of encode"""
        x = x.permute(0, 2, 1)
        x = rearrange(x, f"... {self.x_shape} -> ... {self.data_shape}", J=self.J)  # e.g. (B, L, J*C) -> (B, L, J, C)
        mean_shape = self.mean.shape
        assert x.shape[-len(mean_shape) :] == mean_shape, f"Ending shape is not {mean_shape}"
        data = x * self.std + self.mean
        return data


class Hmlvec263EnDecoder(EnDecoderBase):
    def __init__(
        self,
        stats_module,
        stats_name,
        asb_repr=False,
        smooth_motion=False,
        velocity_padding_strategy=REPEAT_LAST_FRAME,  #  val(REPEAT_LAST_FRAME) | val(ZERO_FRAME_AHEAD)
        use_original_edition=False,
    ):
        """Input LJ3, to Hmlvec263, apply gaussian normalization"""
        super().__init__(stats_module, stats_name)
        self.asb_repr = asb_repr  # option in hml3d.convert_motion_to_hmlvec263
        self.mean = self.mean.reshape(263)
        self.std = self.std.reshape(263)
        self.smooth_motion = smooth_motion
        self.velocity_padding_strategy = velocity_padding_strategy
        self.use_original_edition = use_original_edition

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, J, 3), in ayfz coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, 263, L)
        """
        # FIXME: discuss with XY,
        # 1. why naively usage with 0-padding results in NaN, the number NaN does not match the number of 0-padding
        # 2. what's the best way to handle this? here I just pad the last frame
        data = data.clone()
        if length is not None:  # Handle NaN in hmlvec263 by padding last frame
            for i, l in enumerate(length):
                data[i, l:] = data[i, l - 1]
        hmlvec263 = convert_motion_to_hmlvec263(
            data,
            # seq_len=length, # enable this line to erase the invalid frames
            return_abs=self.asb_repr,
            smooth_motion=self.smooth_motion,
            velocity_padding_strategy=self.velocity_padding_strategy,
            use_original_edition=self.use_original_edition,
        ).transpose(
            -1, -2
        )  # (B, L, 263)
        x = (hmlvec263 - self.mean) / self.std

        # DEBUG
        # from hmr4d.utils.wis3d_utils import make_wis3d, add_joints22_motion_as_lines
        # wis3d = make_wis3d(name='debug')
        # add_joints22_motion_as_lines(data[0], wis3d, name='batch0')

        return x.permute(0, 2, 1)

    def decode(self, x):
        """
        Args:
            x: (B, 263, L)
        Returns:
            data: (B, L, J, 3), in ayfz coordinate
        """
        x = x.permute(0, 2, 1)
        hmlvec263 = x * self.std + self.mean
        data = convert_hmlvec263_to_motion(hmlvec263.transpose(-1, -2), abs_3d=self.asb_repr)
        return data


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


class GMDEmphaProjEnDecoder(Hmlvec263EnDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Projection
        from hmr4d.network.gmd.statistics import GMD_EMPH_PROJ

        emph_proj = torch.from_numpy(GMD_EMPH_PROJ).float()
        self.register_buffer("emph_proj", emph_proj, False)
        self.register_buffer("inv_emph_proj", emph_proj.inverse(), False)

    def encode(self, motion, length=None):
        x = super().encode(motion, length)
        x = einsum(x, self.emph_proj, "b d l , d c -> b c l")
        return x

    def decode(self, x):
        x = einsum(x, self.inv_emph_proj, "b d l , d c -> b c l")
        return super().decode(x)
