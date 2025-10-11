import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds

from hmr4d.network.base_arch.transformer.dit_release import (
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
)
from hmr4d.network.base_arch.transformer.dit_release_mod import DiTBlock
from hmr4d.network.base_arch.transformer_layer.dit_layerv2 import ResCondBlock, zero_module

from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.supermotion.bertlike import length_to_mask
from timm.models.vision_transformer import Mlp


class NetworkDitV1(nn.Module):
    def __init__(
        self,
        # x
        input_dim=218,
        max_len=120,
        # condition
        imgseq_dim=1024,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        noisyobs_dim=126,
        condition_design="v1",
        # intermediate
        latent_dim=512,
        num_layers=8,
        num_heads=4,
        # output
        extra_output_dim=0,
        # training
        dropout=0.1,
    ):
        super().__init__()

        # input
        self.input_dim = input_dim
        self.max_len = max_len

        # condition
        self.imgseq_dim = imgseq_dim
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.noisyobs_dim = noisyobs_dim
        self.condition_design = condition_design

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)
        self.t_embedder = TimestepEmbedder(self.latent_dim)
        self._build_condition_embedder()

        # Will use fixed sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.latent_dim, torch.arange(self.max_len))
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), False)

        # Transformer
        mlp_ratio = 4.0
        self.blocks = nn.ModuleList(
            [DiTBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(self.num_layers)]
        )

        # Output
        self.final_layer = Mlp(self.latent_dim, out_features=self.input_dim)
        self.extra_output = extra_output_dim > 0
        if self.extra_output:
            self.extra_output = Mlp(self.latent_dim + 6, out_features=extra_output_dim)
            self.register_buffer("init_cam", torch.tensor([0.9, 0, 0]), False)

    def _build_condition_embedder(self):
        condition_design = self.condition_design
        latent_dim = self.latent_dim
        dropout = self.dropout

        if self.cliffcam_dim > 0:
            if condition_design == "v1":
                self.cliffcam_embedder = nn.Sequential(
                    nn.Linear(self.cliffcam_dim, latent_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    zero_module(nn.Linear(latent_dim, latent_dim)),
                )

        if self.cam_angvel_dim > 0:
            if condition_design == "v1":
                self.cam_angvel_embedder = nn.Sequential(
                    nn.Linear(self.cam_angvel_dim, latent_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    zero_module(nn.Linear(latent_dim, latent_dim)),
                )

        if self.noisyobs_dim > 0:
            if condition_design == "v1":
                # Map noisyobs to latent_dim, Fuse with timestep (boosting the accel)
                self.embed_noisyobs = nn.Sequential(
                    nn.Linear(self.noisyobs_dim, self.latent_dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim),
                )

                self.noisyobs_block = ResCondBlock(self.latent_dim, self.latent_dim, self.dropout)

        if self.imgseq_dim > 0:
            if condition_design == "v1":
                # Map imgseq to latent_dim, Fuse with x (boosting training)
                self.imgseq_embedder = nn.Sequential(
                    nn.LayerNorm(self.imgseq_dim),
                    nn.Linear(self.imgseq_dim, self.latent_dim),
                )
                self.img_block = ResCondBlock(self.latent_dim, self.latent_dim, self.dropout)

    def forward(self, x, timesteps, length, f_imgseq=None, f_cliffcam=None, f_noisyobs=None, f_cam_angvel=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), nosiy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        # Input
        x = self.x_embedder(x.permute(0, 2, 1))  # (B, L, D)
        t = self.t_embedder(timesteps)  # (B, D)
        t_ = t[:, None].expand(-1, L, -1).contiguous()  # (B, L, D)

        # Condition
        f_to_add = []
        condition_design = self.condition_design
        if f_cliffcam is not None:
            if condition_design == "v1":
                f_delta = self.cliffcam_embedder(f_cliffcam)
            f_to_add.append(f_delta)
        if f_cam_angvel is not None:
            if condition_design == "v1":
                f_delta = self.cam_angvel_embedder(f_cam_angvel)
            f_to_add.append(f_delta)
        if f_noisyobs is not None:
            if condition_design == "v1":
                f_delta = self.embed_noisyobs(f_noisyobs)
                f_delta = self.noisyobs_block.forward_cond(t_, f_delta)
            f_to_add.append(f_delta)
        if f_imgseq is not None:
            if condition_design == "v1":
                f_delta = self.imgseq_embedder(f_imgseq)
                f_delta = self.img_block.forward_cond(x, f_delta)
            f_to_add.append(f_delta)

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        # Transformer
        x = x + self.pos_embed  # (B, L, D)
        for block in self.blocks:
            x = block(x, t, pmask)

        # Output
        sample = self.final_layer(x).permute(0, 2, 1)  # (B, C, L)
        mask = ~pmask.unsqueeze(1)  # (B, 1, L)
        extra_output = None
        if self.extra_output:
            x_ = torch.cat([x, self.init_cam[None, None].expand(B, L, -1), f_cliffcam], dim=-1)  # similar to CLIFF
            pred_cam = self.extra_output(x_)
            pred_cam = torch.cat(
                [
                    (2 * torch.sigmoid(pred_cam[..., :1]) - 1) * self.init_cam[0],  # scale  (not negative)
                    pred_cam[..., 1:3],
                ],
                dim=-1,
            )
            extra_output = (pred_cam + self.init_cam).permute(0, 2, 1)  # (B, C', L)

        return MdmUnetOutput(sample=sample, mask=mask, extra_output=extra_output)


# Add to MainStore

cfg_dit_v1_S = builds(
    NetworkDitV1,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="dit_v1_S", node=cfg_dit_v1_S, group=f"network/supermotion")

cfg_dit_v1_S_extraout3 = cfg_dit_v1_S(extra_output_dim=3)
MainStore.store(name="dit_v1_S_extraout3", node=cfg_dit_v1_S_extraout3, group=f"network/supermotion")

cfg_dit_v1_B = cfg_dit_v1_S(latent_dim=768, num_layers=12, num_heads=8)
MainStore.store(name="dit_v1_B", node=cfg_dit_v1_B, group=f"network/supermotion")

cfg_dit_v1_S_vecv6_extraout3 = cfg_dit_v1_S(input_dim=155, extra_output_dim=3)
MainStore.store(name="dit_v1_S_vecv6_extraout3", node=cfg_dit_v1_S_vecv6_extraout3, group=f"network/supermotion")
