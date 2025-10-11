# https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/attention.py

import torch
from typing import Optional, Callable, Union, List
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, MultiheadAttention
import torch.nn.functional as F


GN_GROUPS = 32


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim_in, inner_dim, glu=True, dropout=0.0):
        super().__init__()
        project_in = nn.Sequential(nn.Linear(dim_in, inner_dim), nn.GELU()) if not glu else GEGLU(dim_in, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_in))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self._ff_block = FeedForward(d_model, dim_feedforward, dropout=dropout)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: List[Tensor],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: List[Optional[Tensor]] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: List[Optional[Tensor]] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            raise NotImplementedError
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.dropout2(x)


class BLCGroupNorm(nn.GroupNorm):
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, use_scale_shift_norm=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # SD uses GroupNorm(32, c)
        self.in_layer = nn.Sequential(
            BLCGroupNorm(GN_GROUPS, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        emb_out_channels = 2 * channels if use_scale_shift_norm else channels
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, emb_out_channels),
        )
        self.out_layer = nn.Sequential(
            BLCGroupNorm(GN_GROUPS, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(channels, channels)),
        )

    def forward(self, x, emb):
        h = self.in_layer(x)
        emb_out = self.emb_layer(emb)
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=-1)
            out_norm, out_rest = self.out_layer[0], self.out_layer[1:]
            h = modulate(out_norm(h), shift, scale)
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layer(h)
        return x + h


class ResCondBlock(nn.Module):
    # Same as ResBlock, always residual add
    def __init__(self, channels, cond_channels, dropout):
        super().__init__()
        self.channels = channels
        self.emb_channels = cond_channels
        self.dropout = dropout

        # SD uses GroupNorm(32, c)
        self.in_layer = nn.Sequential(
            BLCGroupNorm(GN_GROUPS, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.cond_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, channels),
        )
        self.out_layer = nn.Sequential(
            BLCGroupNorm(GN_GROUPS, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(channels, channels)),
        )

    def forward(self, x, cond):
        h = self.in_layer(x)
        cond_out = self.cond_layer(cond)
        h = h + cond_out
        h = self.out_layer(h)
        return x + h


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x, shift, scale):
    assert len(x.shape) == len(shift.shape) == len(scale.shape)
    return x * (1 + scale) + shift
