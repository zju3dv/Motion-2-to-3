import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.mas.mdm import length_to_mask, TimestepEmbedder
from hmr4d.network.base_arch.transformer_layer.encoder_mv import MVTransformerEncoderLayer
from hmr4d.network.base_arch.transformer_layer.decoder_mv import MVTransformerDecoderLayer
from hmr4d.network.base_arch.transformer_layer.sd_layer import zero_module
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.configs import MainStore, builds


class MDM2DMV(nn.Module):
    def __init__(
        self,
        # x
        input_dim=44,
        # condition
        text_dim=512,
        cam_dim=5,
        is_2dinput=True,
        # intermediate
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        # training
        dropout=0.1,
        extra_root=False,
        # clip related
        with_projection=True,
        **kargs,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.text_dim = text_dim
        self.cam_dim = cam_dim
        self.is_2dinput = is_2dinput

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dropout = dropout
        self.extra_root = extra_root
        self.with_projection = with_projection

        self._build_model()

    def _build_model(self):
        self._build_input()
        self._build_condition()
        self._build_transformer()
        self._build_output()
        if self.extra_root:
            self.root_process = OutputProcess(2, self.latent_dim)

    def _build_input(self):
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

    def _build_condition(self):
        # TODO: upgrade residual module
        self.embed_text = nn.Linear(self.text_dim, self.latent_dim)
        self.embed_cam = zero_module(nn.Linear(self.cam_dim, self.latent_dim))
        if self.is_2dinput:
            self.embed_cond2d = zero_module(nn.Linear(self.latent_dim, self.latent_dim))

    def _build_transformer(self):
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.with_projection:
            seqTransEncoderLayer = MVTransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            seqTransDecoderLayer = MVTransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation="gelu",
            )
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def _build_output(self):
        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

    def forward(self, x, timesteps, length, f_text=None, f_cam=None, f_cond2d=None, f_text_length=None):
        """
        Args:
            x: (B, V, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x
            f_text: (B, C)
            f_cam: (B, V, C)
            f_cond2d: (B, C, L)
        """
        B, V, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        emb = emb[:, :, None]  # [1, bs, 1, d]
        if f_text is not None:
            f_text = self.embed_text(f_text)  # [bs, d]
            if self.with_projection:
                emb = emb + f_text[None, :, None]  # [1, bs, 1, d]
        if f_cam is not None:
            f_ = self.embed_cam(f_cam)  # [bs, v, d]
            emb = emb + f_[None]  # [1, bs, v, d]

        x = self.input_process(x)  # [seqlen, bs, v, d]

        if f_cond2d is not None:
            assert self.is_2dinput, "The network does not have 2d condition layer!"
            f_ = self.input_process(f_cond2d)  # [seqlen, bs, d]
            f_ = self.embed_cond2d(f_)  # [seqlen, bs, d]
            x = x + f_[:, :, None]  # [seqlen, bs, v, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, v, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, v, d]

        maskseq = length_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        if self.with_projection:
            output_latent = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, v, d]
        else:
            pmask_text = length_to_mask(f_text_length, f_text.size(1))  # (B, 77)
            f_text = f_text.permute(1, 0, 2)  # [77, bs, d]
            output = self.seqTransDecoder(
                tgt=xseq,
                memory=f_text,
                tgt_key_padding_mask=~maskseq,
                memory_key_padding_mask=~pmask_text,
            )  # [seqlen + 1, bs, d]
            output_latent = output[1:]  # [seqlen, bs, d]

        output = self.output_process(output_latent)  # [bs, v, d, seqlen]
        if self.extra_root:
            root = self.root_process(output_latent) # [bs, v, 2, seqlen]
            output = torch.cat([output[..., :-2, :], root], dim=-2)

        return MdmUnetOutput(sample=output, mask=maskseq[:, None, None, 1:])

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, d_model] -> [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        if len(x.shape) == 4:
            # for mv data
            x = x + self.pe[: x.shape[0], None]
        else:
            raise NotImplementedError
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)  # [bs, d, seqlen] -> [seqlen, bs, d]
        elif len(x.shape) == 4:
            x = x.permute(3, 0, 1, 2)  # [bs, v, d, seqlen] -> [seqlen, bs, v, d]
        x = self.poseEmbedding(x)  # [seqlen, bs, v, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, v, d]
        output = output.permute(1, 2, 3, 0)  # [seqlen, bs, v, d] -> [bs, v, d, seqlen]
        return output


# Add to MainStore
cfg_mdmmv_S = builds(
    MDM2DMV,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mdmmv_S", node=cfg_mdmmv_S, group=f"network/mas")

cfg_mdmmv_nr_S = cfg_mdmmv_S(input_dim=46)
MainStore.store(name="mdmmv_nr_S", node=cfg_mdmmv_nr_S, group=f"network/mas")

cfg_mdmmv_r_S = cfg_mdmmv_S(extra_root=True)
MainStore.store(name="mdmmv_r_S", node=cfg_mdmmv_r_S, group=f"network/mas")

cfg_mdmmv_no2d_r_S = cfg_mdmmv_S(extra_root=True, is_2dinput=False)
MainStore.store(name="mdmmv_no2d_r_S", node=cfg_mdmmv_no2d_r_S, group=f"network/mas")

cfg_mdmmv_textonly_nr_S = cfg_mdmmv_S(input_dim=46, is_2dinput=False)
MainStore.store(name="mdmmv_textonly_nr_S", node=cfg_mdmmv_textonly_nr_S, group=f"network/mas")

# Final Use: mdmmv_no2d_r_S