import numpy as np
from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from einops import einsum, rearrange, repeat


class ControlTransformerEncoderLayer(nn.Module):
    r"""Modify original encoderlayer with multi-view attention
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
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
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
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.c_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.c_linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.c_dropout = nn.Dropout(dropout)
        self.c_linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.c_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.c_norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.c_dropout1 = nn.Dropout(dropout)
        self.c_dropout2 = nn.Dropout(dropout)
        self.c_zerolinear = zero_module(nn.Linear(d_model, d_model))

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        L, B, _, C = src.shape
        assert src_mask is None, "Not Implemented this for control attention"

        src1, src2 = src[..., 0, :], src[..., 1, :]

        x1 = src1
        if self.norm_first:
            x1 = x1 + self._sa_block(self.norm1(x1), src_mask, src_key_padding_mask)
            x1 = x1 + self._ff_block(self.norm2(x1))
        else:
            x1 = self.norm1(x1 + self._sa_block(x1, src_mask, src_key_padding_mask))
            x1 = self.norm2(x1 + self._ff_block(x1))

        x2 = src2
        if self.norm_first:
            x2 = x2 + self._csa_block(self.c_norm1(x2), src_mask, src_key_padding_mask)
            x2 = x2 + self._cff_block(self.c_norm2(x2))
        else:
            x2 = self.c_norm1(x2 + self._csa_block(x2, src_mask, src_key_padding_mask))
            x2 = self.c_norm2(x2 + self._cff_block(x2))

        x1 = x1 + self.c_zerolinear(x2)

        x = torch.stack((x1, x2), dim=-2)

        return x

    def _csa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.c_self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.c_dropout1(x)

    def _cff_block(self, x: Tensor) -> Tensor:
        x = self.c_linear2(self.c_dropout(self.activation(self.c_linear1(x))))
        return self.c_dropout2(x)

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def freeze(self):
        self.self_attn.eval()
        self.self_attn.requires_grad_(False)
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
        self.dropout1.eval()
        self.dropout1.requires_grad_(False)
        self.dropout2.eval()
        self.dropout2.requires_grad_(False)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
