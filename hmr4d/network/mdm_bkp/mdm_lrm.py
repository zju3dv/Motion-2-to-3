import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.mdm.mdm import MDM, lengths_to_mask, PositionalEncoding, TimestepEmbedder, OutputProcess
from hmr4d.network.mdm.mdm_zero1to3 import MDMZero1to3


class MDMLRM(MDMZero1to3):
    def __init__(self, trans_arch="trans_enc", **kargs):
        self.trans_arch = trans_arch
        super().__init__(**kargs)

    def _build_transformer(self):
        if self.trans_arch == "trans_enc":
            super()._build_transformer()
            self.embed_input = nn.Linear(self.latent_dim * 3, self.latent_dim)
        elif self.trans_arch == "trans_dec":
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        else:
            raise NotImplementedError

        self.spatial_embedding = nn.Parameter(torch.randn(200, 1, self.latent_dim), requires_grad=True)

    def _build_output(self):
        # predict 3d joints
        self.output_process = OutputProcess(
            self.njoints * (self.nfeats + 1), self.latent_dim, self.njoints, self.nfeats
        )

    def forward(self, x, timesteps, prompt_latent, d_T, cond_x, length, uncond=False, **kwargs):
        """
        x: [batch_size, c_input, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        prompt_latent: [bs, 1, d]
        d_T: [bs, d_]
        cond_x: [batch_size, c_input, max_frames], another view of motion
        length: [bs] (int) tensor
        uncond: bool
        """
        # inference t might be a int
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        # inference length is 1 as x has mutliview
        if length.shape[0] != x.shape[0]:
            length = length.expand(x.shape[0])

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        cond_x = self.input_process(cond_x)  # [seqlen, bs, d]

        prompt_latent = torch.cat([prompt_latent, d_T[:, None]], dim=-1)

        prompt_emb, cond_x = self.mask_cond(prompt_latent, cond_x, uncond)
        emb += self.embed_cond(prompt_emb)

        x = self.input_process(x)  # [seqlen, bs, d]
        spatial_emb = self.spatial_embedding.expand(-1, x.shape[1], -1)  # [max_seqlen, bs, d]
        # inference, x has different length
        spatial_emb = spatial_emb[: x.shape[0]]  # [seqlen, bs, d]
        if self.trans_arch == "trans_enc":
            x = torch.cat([x, spatial_emb, cond_x], dim=-1)  # [seqlen, bs, 3*d]
        else:
            x = torch.cat([x, spatial_emb], dim=-1)  # [seqlen, bs, 2*d]
        x = self.embed_input(x)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        maskseq = lengths_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        if self.trans_arch == "trans_enc":
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        elif self.trans_arch == "trans_dec":
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=cond_x,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~maskseq[:, 1:],
            )

        output = output[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, feats, nframes]
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])
