import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.base_arch.transformer_layer.decoder_multi_ca import TransformerDecoderLayerMultiCA
from hmr4d.network.base_arch.embeddings import PosEncoding1D, TimestepEmbedder


class Network(nn.Module):
    def __init__(
        self,
        # input
        njoints=22,
        nfeats=3,
        text_dim=512,
        imgseq_dim=512,
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
        # The model is batch_first, composed of linear layers
        # Input: noisy x
        self.pos_encoder = PosEncoding1D(self.latent_dim, self.dropout)
        self.input_process = InputProcess(self.input_dim, self.latent_dim, self.pos_encoder)  # (B, L, D)

        # Condition: PosEnc, Timestep, Text, ImgSeq
        self.embed_timestep = TimestepEmbedder(self.pos_encoder)
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        self.imgseq_process = ImgSeqProcess(self.imgseq_dim, self.latent_dim, self.pos_encoder)

    def _build_transformer(self):
        TransDecoderLayer = TransformerDecoderLayerMultiCA(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_place="post",
            # norm_place="pre",  # it affects the convergence rate ?
            # norm_place="sandwich",
            num_cond=self.num_cond,  # t, text, imgseq
        )
        self.TransDecoder = nn.TransformerDecoder(TransDecoderLayer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

    def forward(self, x, timesteps, length, f_text, f_text_length, f_imgseq, f_imgseq_fid):
        """
        Args:
            x: (B, C, L), a noisy motion sequence, C=263 when using hmlvec263, C=22*3 when using joint*nfeat
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_text: (B, 77, C)
            f_text_length: (B,)
            f_imgseq: (B, I, C)
            f_imgseq_fid: (B, I)
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        # Input
        x = self.input_process(x)  # (B, L, D), in-proj x and add positional encoding
        f_timesteps = self.embed_timestep(timesteps).unsqueeze(1)  # (B, 1, D)
        f_text = self.embed_text(f_text)  # (B, 77, D)
        f_imgseq, pmask_imgseq = self.imgseq_process(f_imgseq, f_imgseq_fid)  # (B, 1+I, D), (B, 1+I)

        # Setup length and make padding mask
        assert B == length.size(0) == f_text_length.size(0) == f_imgseq_fid.size(0)
        pmask_x = ~length_to_mask(length, L)  # (B, L)
        pmask_text = ~length_to_mask(f_text_length, f_text.size(1))  # (B, 77)

        # Transformer
        output = self.TransDecoder(
            tgt=x,
            memory=[f_timesteps, f_text, f_imgseq],
            tgt_key_padding_mask=pmask_x,
            memory_mask=[None, None, None],
            memory_key_padding_mask=[None, pmask_text, pmask_imgseq],
        )

        # Output
        output = self.output_process(output)  # (B, C, L)
        mask = ~pmask_x.unsqueeze(1)  # (B, 1, L)

        return MdmUnetOutput(sample=output, mask=mask)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, pos_encoder):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pos_encoder = pos_encoder  # (L, D)

        self.input_proj = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, L), the axis of the shape is consistent to the project
        """
        x = self.input_proj(x.permute(0, 2, 1))  # (B, C, L) -> (B, L, D)
        x = self.pos_encoder(x)
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
        output = self.poseFinal(output).permute(0, 2, 1)  # (B, D, L)
        return output


class ImgSeqProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, pos_encoder):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pos_encoder = pos_encoder  # (L, D)
        self.embed_imgseq = nn.Linear(self.input_feats, self.latent_dim)
        self.uncond_imgseq_embedding = nn.Parameter(torch.randn(self.latent_dim), requires_grad=True)

    def forward(self, f_imgseq, f_imgseq_fid):
        """
        Args:
            f_imgseq: (B, I, C)
            f_imgseq_fid: (B, I)
        Returns:
            f_imgseq: (B, 1+I, D)
            pmask: (B, 1+I)
        """
        f_imgseq = self.embed_imgseq(f_imgseq)  # (B, I, D)
        B, I, D = f_imgseq.shape

        # Pick the corresponding positional embedding
        pe_picked = self.pos_encoder.pe[f_imgseq_fid.view(-1)].reshape(B, I, -1)
        f_imgseq = f_imgseq + pe_picked  # (B, I, D)

        # Add unconditional embedding to the first frame
        uncond_embedding = repeat(self.uncond_imgseq_embedding, "d -> b 1 d", b=B)
        f_imgseq = torch.cat([uncond_embedding, f_imgseq], dim=1)  # (B, 1+I, D)

        pmask = f_imgseq_fid == -1  # (B, I)
        pmask = F.pad(pmask, (1, 0), value=False)  # (B, 1+I)

        return f_imgseq, pmask


def length_to_mask(lengths, max_len):
    """
    Returns: (B, max_len)
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
