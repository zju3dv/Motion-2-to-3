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
from hmr4d.network.mas.mdm import length_to_mask
from timm.models.vision_transformer import Mlp


class DiT2D(nn.Module):
    def __init__(
        self,
        # x
        input_dim=44,
        max_len=200,
        # condition
        text_dim=512,
        # intermediate
        latent_dim=512,
        num_layers=8,
        num_heads=4,
        # training
        dropout=0.1,
    ):
        super().__init__()

        # input
        self.input_dim = input_dim
        self.max_len = max_len

        # condition
        self.text_dim = text_dim

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

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout

        if self.text_dim > 0:
            self.text_embedder = nn.Sequential(
                nn.Linear(self.text_dim, latent_dim),
            )

    def forward(self, x, timesteps, length, f_text=None):
        """
        Args:
            x: (B, C, L), a noisy motion sequence
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_text: (B, C)
        """
        B, _, L = x.shape

        # Set timesteps
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        assert len(timesteps) == x.shape[0]

        # Input
        x = self.x_embedder(x.permute(0, 2, 1))  # (B, L, D)
        t = self.t_embedder(timesteps)  # (B, D)

        # Condition
        f_to_add = []
        if f_text is not None:
            f_delta = self.text_embedder(f_text)
            f_to_add.append(f_delta)

        for f_delta in f_to_add:
            t = t + f_delta

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

        return MdmUnetOutput(sample=sample, mask=mask)


# Add to MainStore
cfg_dit_S = builds(
    DiT2D,
    latent_dim=512,
    num_layers=8,
    num_heads=4,
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="dit_S", node=cfg_dit_S, group=f"network/mas")

cfg_dit_nr_S = cfg_dit_S(input_dim=46)
MainStore.store(name="dit_nr_S", node=cfg_dit_nr_S, group=f"network/mas")

cfg_dit_v1_B = cfg_dit_S(latent_dim=768, num_layers=12, num_heads=8)
MainStore.store(name="dit_B", node=cfg_dit_v1_B, group=f"network/mas")
