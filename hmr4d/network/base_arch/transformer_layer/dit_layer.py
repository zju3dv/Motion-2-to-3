# https://github.com/facebookresearch/DiT/blob/main/models.py
# We use adaLN in SD, merging ResBlock and BasicTransformerBLock

import torch
from typing import Optional, Any, Union, Callable, List
from torch import Tensor
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from hmr4d.network.base_arch.transformer_layer.sd_layer import FeedForward, modulate, BLCGroupNorm, GN_GROUPS


class BasicDiTTransformerBlock(nn.Module):
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
        self.img_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self._ff_block = FeedForward(d_model, dim_feedforward, dropout=dropout)

        # Implementation of adaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 12 * d_model, bias=True),
        )

        self.norm_first = norm_first
        self.norm0 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout0 = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        timestep: Tensor,
        f_imgseq: Tensor,
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
        (
            shift_img,
            scale_img,
            gate_img,
            shift_sa,
            scale_sa,
            gate_sa,
            shift_mha,
            scale_mha,
            gate_mha,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(timestep).chunk(12, dim=-1)

        if self.norm_first:
            x = x + gate_img * self._img_block(
                modulate(self.norm0(f_imgseq), shift_img, scale_img),
            )
            x = x + gate_sa * self._sa_block(
                modulate(self.norm1(x), shift_sa, scale_sa),
                tgt_mask,
                tgt_key_padding_mask,
            )
            x = x + gate_mha * self._mha_block(
                modulate(self.norm2(x), shift_mha, scale_mha),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = x + gate_mlp * self._ff_block(
                modulate(self.norm3(x), shift_mlp, scale_mlp),
            )
        else:
            raise NotImplementedError
        return x

    # img feature residual block
    def _img_block(self, f_imgseq: Tensor):
        f_imgseq = self.img_block(f_imgseq)
        return self.dropout0(f_imgseq)

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
