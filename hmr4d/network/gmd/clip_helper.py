import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.utils.pylogger import Log


class TransformerReduce(nn.Module):
    def __init__(self, embed_dim=512, num_head=8, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embed_dim * 4,
        )

    def forward(self, x, S):
        BS, D = x.shape
        assert BS % S == 0
        B = BS // S

        x = x.reshape(B, S, D).transpose(0, 1)  # (S, B, D)
        x_avg = x.mean(dim=0, keepdim=True)  # (1, B, D)

        x = torch.concat([x_avg, x], dim=0)  # (S+1, B, D)

        x = self.transformer(x, x_avg)  # (1, B, D)

        output = F.relu(x + x_avg)  # (1, B, D)
        return output.squeeze(1)  # (B, D)
