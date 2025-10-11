import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, pos_encoder, latent_dim=None, time_embed_dim=None):
        """
        Args:
            pos_encoder: Positional encoding module
            latent_dim: The dimension of the latent space
            time_embed_dim: The dimension of the time embedding
        """
        super().__init__()
        self.pos_encoder = pos_encoder

        # Set dims
        if latent_dim is None:  # Use pos_encoder's latent_dim
            self.latent_dim = pos_encoder.d_model
        else:
            self.latent_dim = latent_dim

        if time_embed_dim is None:  # Use latent_dim
            time_embed_dim = self.latent_dim

        # Set networks
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps Tensor: (B,) The timesteps to embed,
        Returns:
            (B, D)
        """
        return self.time_embed(self.pos_encoder.pe[timesteps])
