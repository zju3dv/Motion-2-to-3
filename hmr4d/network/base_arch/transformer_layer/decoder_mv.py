import numpy as np
from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from einops import einsum, rearrange, repeat


class MVTransformerDecoderLayer(nn.Module):
    r"""Modify original decoderlayer with multi-view attention
        Add mv_norm -> mv_attn -> mv_dropout -> mv_linear
        Enable mv_linear with zero initialization

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.mv_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.mv_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.mv_dropout = nn.Dropout(dropout)
        self.mv_linear = zero_module(nn.Linear(d_model, d_model))

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
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

        L, B, V, C = tgt.shape
        assert tgt_mask is None, "Not Implemented this for multi-view attention"
        tgt_key_padding_mask = repeat(tgt_key_padding_mask, "b l -> b v l", v=V)  # (B, V, L)
        tgt_key_padding_mask = tgt_key_padding_mask.reshape(B * V, L)  # (B*V, L)

        memory = repeat(memory, "l b c -> l b v c", v=V)
        mem_L = memory.shape[0]
        memory = memory.reshape(mem_L, B * V, C)
        memory_key_padding_mask = repeat(memory_key_padding_mask, "b l -> b v l", v=V)  # (B, V, L2)
        memory_key_padding_mask = memory_key_padding_mask.reshape(B * V, mem_L)  # (B*V, L2)

        x = tgt
        if self.norm_first:
            x = x + self._mv_attn_block(self.mv_norm(x), tgt_mask)
            x = rearrange(x, "l b v c -> l (b v) c")
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = x + self.mv_norm(self._mv_attn_block(x, tgt_mask))
            x = rearrange(x, "l b v c -> l (b v) c")
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        x = rearrange(x, "l (b v) c -> l b v c", b=B, v=V, c=C)

        return x

    # mv-attention block, attention on view-axis
    def _mv_attn_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        L, B, V, _ = x.shape
        x = rearrange(x, "l b v c -> v (l b) c")
        x = self.mv_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = rearrange(x, "v (l b) c -> l b v c", l=L, b=B, v=V)
        return self.mv_linear(self.mv_dropout(x))

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

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def freeze(self):
        self.self_attn.eval()
        self.self_attn.requires_grad_(False)
        self.multihead_attn.eval()
        self.multihead_attn.requires_grad_(False)
        self.linear1.eval()
        self.linear1.requires_grad_(False)
        self.dropout.eval()
        self.dropout.requires_grad_(False)
        self.linear2.eval()
        self.linear2.requires_grad_(False)
        self.norm1.eval()
        self.norm1.requires_grad_(False)
        self.norm2.eval()
        self.norm2.requires_grad_(False)
        self.norm3.eval()
        self.norm3.requires_grad_(False)
        self.dropout1.eval()
        self.dropout1.requires_grad_(False)
        self.dropout2.eval()
        self.dropout2.requires_grad_(False)
        self.dropout3.eval()
        self.dropout3.requires_grad_(False)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
