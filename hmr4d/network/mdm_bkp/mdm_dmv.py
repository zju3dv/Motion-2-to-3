import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.mdm.mdm import (
    MDM,
    lengths_to_mask,
    PositionalEncoding,
    TimestepEmbedder,
    OutputProcess,
    InputProcess,
)
from hmr4d.network.mdm.mdm_zero1to3 import MDMZero1to3
from hmr4d.network.mdm.mdm_lrm import MDMLRM
from einops import einsum, rearrange, repeat


class MDMDMV(MDM):
    def __init__(self, trans_arch="trans_enc", **kargs):
        self.trans_arch = trans_arch
        super().__init__(**kargs)

    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.trans_arch == "trans_enc":
            seqTransEncoderLayer = SeqViewTransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        elif self.trans_arch == "trans_dec":
            seqTransDecoderLayer = SeqViewTransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        else:
            raise NotImplementedError

        self.d_T_net = nn.Linear(4, self.latent_dim)

    def forward(self, x, timesteps, prompt_latent, d_T, length, uncond=False, **kwargs):
        """
        x: [batch_size, N, c_input, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        prompt_latent: [bs, 1, d]
        d_T: [bs, N, d_]
        length: [bs] (int) tensor
        uncond: bool
        """
        B, N, _, L = x.shape
        # inference t might be a int
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        # inference length is 1 as x has mutliview
        if length.shape[0] != x.shape[0]:
            length = length.expand(x.shape[0])

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        prompt_emb = self.mask_cond(prompt_latent, uncond)
        emb += self.embed_text(prompt_emb)

        x = rearrange(x, "b n c l -> (b n) c l")
        x = self.input_process(x)  # [seqlen, bs*n, d]
        x = rearrange(x, "l (b n) c -> l b n c", n=N)  # [seqlen, bs, n, d]

        d_T = self.d_T_net(d_T)  # [bs, N, d]
        x = x + d_T  # [seqlen, bs, n, d]

        # adding the timestep embed
        xseq = torch.cat((rearrange(emb, "l b d -> l b 1 d").expand(-1, -1, N, -1), x), dim=0)  # [seqlen+1, bs, n, d]

        xseq = rearrange(xseq, "l b n d -> l (b n) d")
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, n, d]
        xseq = rearrange(xseq, "l (b n) d -> l b n d", n=N)

        mask_seq = lengths_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        mask_xseq = rearrange(mask_seq, "b l -> b 1 l").expand(-1, N, -1)  # [bs, n+1, seqlen+1]
        mask_xseq = rearrange(mask_xseq, "b n l -> (b n) l ")  # [bs*(n+1), seqlen+1]

        if self.trans_arch == "trans_enc":
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~mask_xseq)
        else:
            raise NotImplementedError

        output = output[1:]  # [seqlen, bs, n, d]

        output = self.output_process(output)  # [bs, n, d, seqlen]
        return MdmUnetOutput(sample=output, mask=mask_seq[:, None, 1:])


class MDMDMV3D(MDMDMV):
    def _build_transformer(self):
        super()._build_transformer()
        self.spatial_embedding = nn.Parameter(torch.randn(200, 1, self.latent_dim), requires_grad=True)

    def _build_output(self):
        # predict 3d joints
        self.output_process = OutputProcess(
            self.njoints * (self.nfeats + 1), self.latent_dim, self.njoints, self.nfeats
        )

    def forward(self, x, timesteps, prompt_latent, d_T, length, uncond=False, **kwargs):
        """
        x: [batch_size, N, c_input, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        prompt_latent: [bs, 1, d]
        d_T: [bs, N, d_]
        length: [bs] (int) tensor
        uncond: bool
        """
        B, N, _, L = x.shape
        # inference t might be a int
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        # inference length is 1 as x has mutliview
        if length.shape[0] != x.shape[0]:
            length = length.expand(x.shape[0])

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        prompt_emb = self.mask_cond(prompt_latent, uncond)
        emb += self.embed_text(prompt_emb)

        x = rearrange(x, "b n c l -> (b n) c l")
        x = self.input_process(x)  # [seqlen, bs*n, d]
        x = rearrange(x, "l (b n) c -> l b n c", n=N)  # [seqlen, bs, n, d]
        spatial_emb = self.spatial_embedding.expand(-1, B, -1)  # [max_seqlen, bs, d]
        # inference, x has different length
        spatial_emb = spatial_emb[:L]  # [seqlen, bs, d]

        d_T = self.d_T_net(d_T[..., :N, :])  # [bs, N, d]
        x = x + d_T  # [seqlen, bs, n, d]

        # adding the timestep embed
        xseq = torch.cat((rearrange(emb, "l b d -> l b 1 d").expand(-1, -1, N, -1), x), dim=0)  # [seqlen+1, bs, n, d]
        spatialseq = torch.cat((emb, spatial_emb), dim=0)  # [seqlen+1, bs, d]

        xseq = rearrange(xseq, "l b n d -> l (b n) d")
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, n, d]
        xseq = rearrange(xseq, "l (b n) d -> l b n d", n=N)
        spatialseq = self.sequence_pos_encoder(spatialseq)  # [seqlen+1, bs, d]
        spatialseq = rearrange(spatialseq, "l b d -> l b 1 d")

        mask_spatialseq = lengths_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        mask_xseq = rearrange(mask_spatialseq, "b l -> b 1 l").expand(-1, N + 1, -1)  # [bs, n+1, seqlen+1]
        mask_xseq = rearrange(mask_xseq, "b n l -> (b n) l ")  # [bs*(n+1), seqlen+1]

        xseq = torch.cat([spatialseq, xseq], dim=-2)  # [seqlen+1, bs, n + 1, d]

        if self.trans_arch == "trans_enc":
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~mask_xseq)
        elif self.trans_arch == "trans_dec":
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=None,  # contain memory in xseq
                tgt_key_padding_mask=~mask_xseq,
                memory_key_padding_mask=~mask_xseq,
            )

        output = output[1:, :, 0]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, feats, nframes]
        return MdmUnetOutput(sample=output, mask=mask_spatialseq[:, None, 1:])


class SeqViewTransformerEncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.viewTransEncoderLayer = nn.TransformerEncoderLayer(*args, **kwargs)
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """_summary_

        Args:
            x (_type_): L, B, N + 1, C

        Returns:
            _type_: _description_
        """
        L, B, _, _ = x.shape
        x = rearrange(x, "l b n c -> n (l b) c")
        x = self.viewTransEncoderLayer(x)
        x = rearrange(x, "n (l b) c -> l (b n) c", l=L, b=B)
        x = self.seqTransEncoderLayer(x, *args, **kwargs)
        x = rearrange(x, "l (b n) c -> l b n c", b=B)
        return x


class SeqViewTransformerDecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.viewTransDecoderLayer = nn.TransformerDecoderLayer(*args, **kwargs)
        self.seqTransDecoderLayer = nn.TransformerDecoderLayer(*args, **kwargs)

    def forward(self, tgt, memory, *args, **kwargs):
        """_summary_

        Args:
            x (_type_): L, B, N + 1, C
            memory: None, has it in tgt

        Returns:
            _type_: _description_
        """
        x = tgt
        L, B, _, _ = x.shape
        spatialseq = x[..., :1, :]  # [L, B, 1, C]
        spatialseq = rearrange(spatialseq, "l b 1 c -> 1 (l b) c")
        view_mem = rearrange(x[..., 1:, :], "l b n c -> n (l b) c")
        output = self.viewTransDecoderLayer(tgt=spatialseq, memory=view_mem)  # [1, seqlen*bs, d]
        x = torch.cat([output, view_mem], dim=0)  # [N + 1, seqlen*bs, d]
        x = rearrange(x, "n (l b) c -> l (b n) c", l=L)
        x = self.seqTransDecoderLayer(tgt=x, memory=x, *args, **kwargs)
        x = rearrange(x, "l (b n) c -> l b n c", b=B)
        return x
