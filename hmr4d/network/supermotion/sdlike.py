import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from torch.nn.modules.transformer import TransformerDecoder
from hmr4d.network.base_arch.transformer_layer.sd_layer import (
    BasicTransformerBlock,
    GN_GROUPS,
    ResBlock,
    BLCGroupNorm,
    ResCondBlock,
)
from hmr4d.network.base_arch.transformer_layer.dit_layer import BasicDiTTransformerBlock
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.base_arch.embeddings import PosEncoding1D, TimestepEmbedder
from hmr4d.network.supermotion.bertlike import InputProcess, OutputProcess, ImgSeqProcess, length_to_mask


class Network(nn.Module):
    def __init__(
        self,
        # input
        njoints=22,
        nfeats=3,
        text_dim=512,
        imgseq_dim=512,
        cam_dim=0,
        max_len=200,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=[2, 2, 2],
        num_heads=4,
        # training
        dropout=0.1,
    ):
        super().__init__()

        # input
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_dim = self.njoints * self.nfeats
        self.text_dim = text_dim
        self.imgseq_dim = imgseq_dim
        self.cam_dim = cam_dim
        self.max_len = max_len

        # intermediate
        self.num_cond = 3  # t, text, imgseq
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        if isinstance(num_layers, int):
            num_layers = [num_layers]
        self.num_layers = num_layers
        self.num_stages = len(num_layers)
        self.num_heads = num_heads
        self.dropout = dropout

        # build model
        self._build_input()
        self._build_transformer()
        self._build_output()

    def _build_input(self):
        # Global positional encoding
        self.sincos_pos_embedding = PosEncoding1D(self.latent_dim, self.dropout)

        # The model is batch_first, composed of linear layers
        # Input: noisy x
        max_len = self.max_len
        self.learned_pos_embedding = nn.Parameter(torch.randn(max_len, self.latent_dim), requires_grad=True)
        self.input_process = InputProcess(self.input_dim, self.latent_dim, self.learned_pos_embedding)

        # Condition: PosEnc, Timestep, Text, ImgSeq
        self.embed_timestep = TimestepEmbedder(self.sincos_pos_embedding)
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        self.imgseq_process = ImgSeqProcess(self.imgseq_dim, self.latent_dim, self.learned_pos_embedding)
        if self.cam_dim > 0:
            self.embed_camext = nn.Sequential(
                nn.Linear(4, self.latent_dim * 2),
                nn.SiLU(),
                nn.Linear(self.latent_dim * 2, self.latent_dim),
            )

    def _build_transformer(self):
        layer = BasicTransformerBlock(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,  # prenorm
        )
        for i in range(self.num_stages):
            res_block = ResBlock(self.latent_dim, self.latent_dim, self.dropout)
            res_cond_block = ResCondBlock(self.latent_dim, self.latent_dim, self.dropout)

            transformer_block = TransformerDecoder(layer, num_layers=self.num_layers[i])
            block = TimestepEmbedSequential(
                res_block,
                BLCGroupNorm(GN_GROUPS, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                res_cond_block,
                BLCGroupNorm(GN_GROUPS, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                transformer_block,
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            setattr(self, f"block_{i}", block)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

    def forward(self, x, timesteps, length, f_text, f_text_length, f_imgseq, f_camext=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence, C=263 when using hmlvec263, C=22*3 when using joint*nfeat
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_text: (B, 77, C)
            f_text_length: (B,)
            f_imgseq: (B, I, C)
            f_camext: (B, 1, 4), camera extrinsic parameters
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        # Input
        f_motion = self.input_process(x)  # (B, 200, D), in-proj x and add positional encoding
        f_timesteps = self.embed_timestep(timesteps).unsqueeze(1)  # (B, 1, D)
        if f_camext is not None:
            f_camext = self.embed_camext(f_camext)  # (B, 1, D)
            f_timesteps = f_timesteps + f_camext  # (B, 1, D)
        f_text = self.embed_text(f_text)  # (B, 77, D)
        f_imgseq = self.imgseq_process(f_imgseq)  # (B, I, D)

        # Setup length and make padding mask
        assert B == length.size(0) == f_text_length.size(0)
        pmask_motion = ~length_to_mask(length, f_motion.size(1))  # (B, L)
        pmask_text = ~length_to_mask(f_text_length, f_text.size(1))  # (B, 77)

        xseq = self.sincos_pos_embedding(f_motion)  # (B, L, D)
        for i in range(self.num_stages):
            block = getattr(self, f"block_{i}")
            xseq = block(xseq, f_timesteps, f_text, f_imgseq, pmask_motion, pmask_text)

        output = xseq[:, :L]  # (B, L, C)
        pmask_motion = pmask_motion[:, :L]  # (B, L)

        output = self.output_process(output)  # (B, C, L)
        mask = ~pmask_motion.unsqueeze(1)  # (B, 1, L)

        return MdmUnetOutput(sample=output, mask=mask)


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        f_text: torch.Tensor,
        f_imgseq: torch.Tensor,
        pmask_x: torch.Tensor = None,
        pmask_text: torch.Tensor = None,
    ):
        for layer in self:
            module = layer

            if isinstance(module, ResBlock):
                x = layer(x, timestep)
            elif isinstance(module, ResCondBlock):
                x = layer(x, f_imgseq)
            elif isinstance(module, nn.TransformerDecoder):
                x = layer(
                    tgt=x,
                    memory=f_text,
                    tgt_key_padding_mask=pmask_x,
                    memory_key_padding_mask=pmask_text,
                )
            else:
                x = layer(x)
        return x
