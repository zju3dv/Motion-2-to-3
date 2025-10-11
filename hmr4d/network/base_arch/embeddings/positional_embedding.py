import torch
import torch.nn as nn
import numpy as np


class PosEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, False)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class FreqEncoder:
    """Copied from Haotong Lin's code"""

    default_encoder_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "num_freqs": 10,  # 3 + 3*128= 387
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    def __init__(self, **kwargs):
        self.kwargs = {**self.default_encoder_kwargs, **kwargs}
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        # max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs["num_freqs"]
        max_freq = N_freqs - 1

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
