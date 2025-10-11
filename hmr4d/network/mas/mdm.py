import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from hmr4d.network.base_arch.transformer_layer.encoder_control import ControlTransformerEncoderLayer
from hmr4d.network.base_arch.transformer_layer.decoder_control import ControlTransformerDecoderLayer
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.configs import MainStore, builds


def length_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MDM2D(nn.Module):
    def __init__(
        self,
        # x
        input_dim=44,
        # condition
        text_dim=512,
        cam_dim=0,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        # training
        dropout=0.1,
        # clip related
        with_projection=True,
        **kargs,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.text_dim = text_dim
        self.cam_dim = cam_dim

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dropout = dropout
        self.with_projection = with_projection

        self._build_model()

    def _build_model(self):
        self._build_input()
        self._build_condition()
        self._build_transformer()
        self._build_output() 

    def _build_input(self):
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

    def _build_condition(self):
        # TODO: upgrade residual module
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        if self.cam_dim > 0:
            self.embed_cam = zero_module(nn.Linear(self.cam_dim, self.latent_dim))

    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.with_projection:
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

    def forward(self, x, timesteps, length, f_text=None, f_cam=None, f_text_length=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x
            f_text: (B, C)
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if f_text is not None:
            f_text = self.embed_text(f_text)
            if self.with_projection:
                emb = emb + f_text[None]  # [1, bs, d]
        if f_cam is not None:
            f_ = self.embed_cam(f_cam)  # [bs, d]
            emb = emb + f_[None]  # [1, bs, d]

        x = self.input_process(x)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        maskseq = length_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        if self.with_projection:
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, d]
        else:
            pmask_text = length_to_mask(f_text_length, f_text.size(1))  # (B, 77)
            f_text = f_text.permute(1, 0, 2)  # [77, bs, d]
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=f_text,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~pmask_text,
            )  # [seqlen + 1, bs, d]
            output = output[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, d, seqlen]
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])

    def freeze(self):
        if self.with_projection:
            self.seqTransEncoder.eval()
            self.seqTransEncoder.requires_grad_(False)
        else:
            self.seqTransDecoder.eval()
            self.seqTransDecoder.requires_grad_(False)
        self.input_process.eval()
        self.input_process.requires_grad_(False)
        self.embed_timestep.eval()
        self.embed_timestep.requires_grad_(False)
        self.embed_text.eval()
        self.embed_text.requires_grad_(False)
        self.output_process.eval()
        self.output_process.requires_grad_(False)

    def freeze_backbone(self):
        if self.with_projection:
            self.seqTransEncoder.eval()
            self.seqTransEncoder.requires_grad_(False)
        else:
            self.seqTransDecoder.eval()
            self.seqTransDecoder.requires_grad_(False)
        self.input_process.eval()
        self.input_process.requires_grad_(False)
        self.embed_timestep.eval()
        self.embed_timestep.requires_grad_(False)
        self.output_process.eval()
        self.output_process.requires_grad_(False)


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
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.permute(1, 2, 0)  # [seqlen, bs, d] -> [bs, d, seqlen]
        return output


# Add to MainStore
cfg_mdm_S = builds(
    MDM2D,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdm_S", node=cfg_mdm_S, group=f"network/mas")

cfg_mdm_d_S = cfg_mdm_S(dropout=0.0, num_heads=8)
MainStore.store(name="mdm_S_drop0", node=cfg_mdm_d_S, group=f"network/mas")

cfg_mdm_nr_S = cfg_mdm_S(input_dim=46)
MainStore.store(name="mdm_nr_S", node=cfg_mdm_nr_S, group=f"network/mas")


class CMDM2D(MDM2D):
    def __init__(self, input_dim=44, text_dim=512, cam_dim=0, latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, with_projection=True, extra_root=False, **kargs):
        super().__init__(input_dim, text_dim, cam_dim, latent_dim, ff_size, num_layers, num_heads, dropout, with_projection, **kargs)
        self.extra_root = extra_root
        if extra_root:
            self.root_process = OutputProcess(2, self.latent_dim)


    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.with_projection:
            seqTransEncoderLayer = ControlTransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            seqTransDecoderLayer = ControlTransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def forward(self, x, timesteps, length, f_text=None, f_cam=None, f_text_length=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x
            f_text: (B, C)
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        emb1 = emb.clone()
        if f_text is not None:
            f_text = self.embed_text(f_text)
            if self.with_projection:
                emb1 = emb1 + f_text[None]  # [1, bs, d]

        emb2 = emb.clone()
        if f_cam is not None:
            f_ = self.embed_cam(f_cam)  # [bs, d]
            emb2 = emb2 + f_[None]  # [1, bs, d]

        x = self.input_process(x)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq1 = torch.cat((emb1, x), dim=0)  # [seqlen+1, bs, d]
        xseq1 = self.sequence_pos_encoder(xseq1)  # [seqlen+1, bs, d]

        xseq2 = torch.cat((emb2, x), dim=0)  # [seqlen+1, bs, d]
        xseq2 = self.sequence_pos_encoder(xseq2)  # [seqlen+1, bs, d]

        maskseq = length_to_mask(length + 1, xseq1.shape[0])  # [bs, seqlen+1]
        xseq = torch.stack([xseq1, xseq2], dim=-2)  # [seqlen+1, bs, 2, d]
        if self.with_projection:
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, 2, d]
        else:
            pmask_text = length_to_mask(f_text_length, f_text.size(1))  # (B, 77)
            f_text = f_text.permute(1, 0, 2)  # [77, bs, d]
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=f_text,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~pmask_text,
            )  # [seqlen + 1, bs, d]
            output = output[1:]  # [seqlen, bs, d]

        output_latent = output[..., 0, :]  # [seqlen, bs, d]

        output = self.output_process(output_latent)  # [bs, d, seqlen]
        if self.extra_root:
            root = self.root_process(output_latent)
            output = torch.cat([output[:, :-2, :], root], dim=1)

        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])

    def freeze(self):
        if self.with_projection:
            for layer in self.seqTransEncoder.layers:
                layer.freeze()
        else:
            for layer in self.seqTransDecoder.layers:
                layer.freeze()
        self.input_process.eval()
        self.input_process.requires_grad_(False)
        self.embed_timestep.eval()
        self.embed_timestep.requires_grad_(False)
        self.embed_text.eval()
        self.embed_text.requires_grad_(False)
        self.output_process.eval()
        self.output_process.requires_grad_(False)


# Add to MainStore
cfg_cmdm_S = builds(
    CMDM2D,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="cmdm_S", node=cfg_cmdm_S, group=f"network/mas")

cfg_cmdm_d_S = cfg_cmdm_S(dropout=0.0, num_heads=8)
MainStore.store(name="cmdm_d_S", node=cfg_cmdm_d_S, group=f"network/mas")

cfg_cmdm_r_S = cfg_cmdm_S(extra_root=True)
MainStore.store(name="cmdm_r_S", node=cfg_cmdm_r_S, group=f"network/mas")

cfg_cmdm_nr_S = cfg_cmdm_S(input_dim=46)
MainStore.store(name="cmdm_nr_S", node=cfg_cmdm_nr_S, group=f"network/mas")
