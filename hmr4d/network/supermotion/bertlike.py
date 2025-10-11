import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.base_arch.embeddings import PosEncoding1D, TimestepEmbedder


class Network(nn.Module):
    def __init__(
        self,
        # input
        njoints=22,
        nfeats=3,
        text_dim=512,
        imgseq_dim=1024,
        cam_dim=0,
        max_len=200,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        activation="gelu",
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

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
        layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True,  # prenorm
        )
        self.TransEncoder = TransformerEncoder(layer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

    def forward(self, x, timesteps, length, f_text, f_text_length, f_imgseq, f_camext):
        """
        Args:
            x: (B, C, L), a noisy motion sequence, C=263 when using hmlvec263, C=22*3 when using joint*nfeat
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_text: (B, 77, C)
            f_text_length: (B,)
            f_imgseq: (B, I, C) I = L
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
        pmask_timestep = pmask_motion.new([False]).expand(B, 1)  # (B, 1)
        pmask_text = ~length_to_mask(f_text_length, f_text.size(1))  # (B, 77)

        # Simply add img conditions as they are aligned, concate causes too large GPU memory
        f_motion = f_motion + f_imgseq

        # Transformer
        x = torch.cat([f_motion, f_timesteps, f_text], dim=1)  # (B, L'=200+1+77, D)
        x = self.sincos_pos_embedding(x)  # (B, L', D)
        pmask_x = torch.cat([pmask_motion, pmask_timestep, pmask_text], dim=1)  # (B, L')
        output = self.TransEncoder(src=x, src_key_padding_mask=pmask_x)  # (B, L', D)
        output = output[:, :L]  # (B, L, D)
        pmask_x = pmask_x[:, :L]  # (B, L)

        # Output
        output = self.output_process(output)  # (B, C, L)
        mask = ~pmask_x.unsqueeze(1)  # (B, 1, L)

        return MdmUnetOutput(sample=output, mask=mask)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, learned_pos_embedding):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.learned_pos_embedding = learned_pos_embedding  # (L, D)
        self.max_len = learned_pos_embedding.shape[0]
        self.input_proj = nn.Linear(self.input_feats + self.latent_dim, self.latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, L), the axis of the shape is consistent to the project
        """
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = F.pad(x, (0, 0, 0, self.max_len - x.shape[1]))  # (B, 200, C)
        pe = repeat(self.learned_pos_embedding, "l d -> b l d", b=x.shape[0])
        x = torch.cat([x, pe], dim=-1)  # (B, L, C+D)
        x = self.input_proj(x)  # (B, L, C+D) -> (B, L, D)
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints=None, nfeats=None):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output).permute(0, 2, 1)  # (B, D, L)
        return output


class ImgSeqProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, learned_pos_embedding):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.learned_pos_embedding = learned_pos_embedding  # (L, D)
        self.max_len = learned_pos_embedding.shape[0]
        self.embed_imgseq = nn.Linear(self.input_feats + self.latent_dim, self.latent_dim)

    def forward(self, f_imgseq):
        """
        Args:
            f_imgseq: (B, I, C)
        Returns:
            f_imgseq: (B, I, D)
        """
        # Pick the corresponding positional embedding
        f_imgseq = F.pad(f_imgseq, (0, 0, 0, self.max_len - f_imgseq.shape[1]))  # (B, 200, C)
        pe = repeat(self.learned_pos_embedding, "l d -> b l d", b=f_imgseq.shape[0])
        x = torch.cat([f_imgseq, pe], dim=-1)  # (B, L, C+D)
        x = self.embed_imgseq(x)  # (B, L, C+D) -> (B, L, D)

        return x


def length_to_mask(lengths, max_len):
    """
    Returns: (B, max_len)
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
