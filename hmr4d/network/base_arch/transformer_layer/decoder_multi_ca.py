import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, MultiheadAttention
from typing import Optional, Callable, Union, List
import copy


class TransformerDecoderLayerMultiCA(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

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
        norm_place: str = "post",  #  ['post', 'pre', 'sandwich']
        device=None,
        dtype=None,
        num_cond: int = 3,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)

        self.num_cond = num_cond
        self.cond_attn = ModuleList(
            [
                MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
                for _ in range(num_cond)
            ]
        )

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_place = norm_place
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        if norm_place == "post":
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        elif norm_place == "pre":
            self.norm2 = ModuleList([LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) for _ in range(num_cond)])
        else:
            assert norm_place == "sandwich"
            self.norm2 = ModuleList(
                [LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) for _ in range(num_cond * 2)]
            )

        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

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
        if self.norm_place == "post":
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

            # change one mha to multi-mha
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            cross_info = 0
            for i in range(self.num_cond):
                cross_info += self._mha_block(i, x, memory[i], memory_mask[i], memory_key_padding_mask[i])
            x = self.norm2(x + cross_info)

            x = self.norm3(x + self._ff_block(x))

        elif self.norm_place == "pre":
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

            # change one mha to multi-mha
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            cross_info = 0
            for i in range(self.num_cond):
                cross_info += self._mha_block(
                    i, self.norm2[i](x), memory[i], memory_mask[i], memory_key_padding_mask[i]
                )
            x = x + cross_info
            x = x + self._ff_block(self.norm3(x))
        else:
            assert self.norm_place == "sandwich"
            raise NotImplementedError
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

            # change one mha to multi-mha
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            cross_info = 0
            for i in range(self.num_cond):
                cross_info += self._mha_block(i, x, memory[i], memory_mask[i], memory_key_padding_mask[i])
            x = self.norm2[0](x + cross_info)

            x = self.norm3(x + self._ff_block(x))

            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

            # change one mha to multi-mha
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            cross_info = 0
            for i in range(self.num_cond):
                cross_info += self._mha_block(i, x, memory[i], memory_mask[i], memory_key_padding_mask[i])
            x = self.norm2[1](x + cross_info)

            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self, i, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        layer = self.cond_attn[i]
        x = layer(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
