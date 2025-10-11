import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MDM(nn.Module):
    def __init__(
        self,
        njoints=22,
        nfeats=2,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        clip_dim=512,
        cond_mask_prob=0.1,
        **kargs
    ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim

        self.input_dim = self.njoints * self.nfeats

        self.cond_mask_prob = cond_mask_prob
        self._build_model()

    def _build_model(self):
        self._build_input()
        self._build_condition()
        self._build_transformer()
        self._build_output()

    def _build_input(self):
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

    def _build_condition(self):
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

    def mask_cond(self, cond, force_mask=False):
        cond = cond.permute(1, 0, 2)  # [bs, 1, d] -> [1, bs, d]
        _, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, prompt_latent=None, length=None, uncond=False, **kwargs):
        """
        x: [batch_size, c_input, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        prompt_latent: [bs, 1, d]
        length: [bs] (int) tensor
        uncond: bool
        """
        # inference t might be a int
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        # inference length is fewer as x has mutliview
        if length is not None and length.shape[0] != x.shape[0]:
            length = length.expand(x.shape[0])
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if prompt_latent is not None:
            emb += self.embed_text(self.mask_cond(prompt_latent, uncond))

        x = self.input_process(x)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        if length is not None:
            maskseq = lengths_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        else:
            maskseq = torch.ones([xseq.shape[1], xseq.shape[0]], dtype=torch.bool, device=xseq.device)  # [bs, seqlen+1]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [bs, d, seqlen] -> [seqlen, bs, d]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        if len(output.shape) == 3:
            output = output.permute(1, 2, 0)  # [seqlen, bs, d] -> [bs, d, seqlen]
        elif len(output.shape) == 4:
            output = output.permute(1, 2, 3, 0)  # [seqlen, bs, n, d] -> [bs, n, d, seqlen]
        else:
            raise NotImplementedError
        return output
