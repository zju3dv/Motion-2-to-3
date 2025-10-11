import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .mdm_unet_utils import PositionalEncoding, TimestepEmbedder
from .mdm_unet_utils import (
    ResidualTemporalBlock,
    Residual,
    Upsample1d,
    PreNorm,
    LinearAttention,
    Downsample1d,
    Conv1dBlock,
)

from hmr4d.network.base_arch.lora import LoRACompatibleLinear, LoRACompatibleConv1d
from hmr4d.utils.pylogger import Log


@dataclass
class MdmUnetOutput(BaseOutput):
    """
    Args: sample (`torch.FloatTensor` of shape `(batch_size, num_channels, num_frames)`)
    """

    sample: torch.FloatTensor = None
    mask: torch.BoolTensor = None
    extra_output: torch.FloatTensor = None


class TemporalUnet(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        adagn=False,
        zero=False,
    ):
        super().__init__()
        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print("dims: ", dims, "mults: ", dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            LoRACompatibleLinear(cond_dim, dim * 4),
            nn.Mish(),
            LoRACompatibleLinear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # use the last dim_in to support the case where the mult doesn't start with 1.
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5),
            LoRACompatibleConv1d(dim_in, input_dim, 1),
        )

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: (B, D_in, F), F means frames or sequence-length in the original code
            cond: (B, D_latent)
        """
        c = self.time_mlp(cond)
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x


class TemporalUnetLarge(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        out_mult=8,
        attention=False,
        adagn=False,
        zero=False,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print("dims: ", dims, "mults: ", dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            LoRACompatibleLinear(cond_dim, dim * 4),
            nn.Mish(),
            LoRACompatibleLinear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # use the last dim_in to support the case where the mult doesn't start with 1.
        final_in = self.cal_concat_multiple(dim_in, input_dim, out_mult)
        # only temporary
        final_type = 4  # NOTE: flag.LARGE_OUT_TYPE
        if final_type == 1:
            print("using final type 1")
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                LoRACompatibleConv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                LoRACompatibleConv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                LoRACompatibleConv1d(out_mult * input_dim, input_dim, 1, groups=input_dim),
            )
        elif final_type == 2:
            # more kernels of size 5
            print("using final type 2")
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                LoRACompatibleConv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                LoRACompatibleConv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                LoRACompatibleConv1d(out_mult * input_dim, input_dim, 5, padding=2, groups=input_dim),
            )
        elif final_type == 3:
            # all kernels of size 5
            print("using final type 3")
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                LoRACompatibleConv1d(dim_in + input_dim, final_in, 5, padding=2),
                # [batch, mult * in_dim, seqlen]
                LoRACompatibleConv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                LoRACompatibleConv1d(out_mult * input_dim, input_dim, 5, padding=2, groups=input_dim),
            )
        else:
            raise NotImplementedError()

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[-1].weight)
            nn.init.zeros_(self.final_conv[-1].bias)

    @staticmethod
    def cal_concat_multiple(in1, in2, multiple):
        """
        calculate the output channels of the concatenation of the two inputs while keeping the output channels a multiple of the given number
        """
        a = (in1 + in2) / multiple
        return int((1 - (a - math.floor(a))) * multiple + in1 + in2)

    def forward(self, x, cond):
        """
        Args:
            x: (B, D_in, F), F means frames or sequence-length in the original code
            cons: (B, D_latent)
        """

        src = x
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        # [batch, last_dim + in_dim, seqlen]
        x = torch.concat([x, src], dim=1)
        # [batch, in_dim, seqlen]
        x = self.final_conv(x)
        # print('x:', x.shape)

        return x


class MdmUnet(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=512,  # D_latnet
        # --- special option in unet --- #
        arch="unet",
        dim_mults=(1, 2, 4, 8),
        attention=False,
        adagn=False,
        zero=False,
        unet_out_mult=8,  # only used in unet_large
        version=None,
        norm_method=None,
        **kwargs,
    ):
        """
        Do not include training tricks, e.g. cond_mask_prob=.1
        Args:
            version: if `GMD_traj_release' then set frames to zero between 196 to 224
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dim_mults = dim_mults
        self.attention = attention

        print("Using UNET with lantent dim: ", self.latent_dim, " and mults: ", self.dim_mults)
        if arch == "unet":
            self.unet = TemporalUnet(
                input_dim=self.input_dim,
                cond_dim=self.latent_dim,
                dim=self.latent_dim,
                dim_mults=self.dim_mults,
                attention=self.attention,
                adagn=adagn,
                zero=zero,
            )
        elif arch == "unet_large":
            print("UNET large variation with output multiplier: ", unet_out_mult)
            self.unet = TemporalUnetLarge(
                input_dim=self.input_dim,
                cond_dim=self.latent_dim,
                dim=self.latent_dim,
                dim_mults=self.dim_mults,
                attention=self.attention,
                adagn=adagn,
                zero=zero,
                out_mult=unet_out_mult,
            )
        else:
            raise NotImplementedError()

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Currently, we can follow the behaviour of the original GMD when handling masking in GMD_traj_release
        self.version = version
        self.norm_method = norm_method

    def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        """
        Args:
            sample: [batch_size, input_dim, num_frames]
            timestep (`torch.FloatTensor`): (B,) or single T The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): (batch, sequence_length, feature_dim)
        Returns: [batch_size, input_dim, num_frames]
        """
        B, D, F = sample.shape

        # 1. time
        if len(timestep.shape) == 0:
            timesteps = timestep.reshape([1]).to(sample.device).expand(sample.shape[0])
        else:
            assert B == timestep.shape[0]
            timesteps = timestep

        emb = self.embed_timestep(timesteps).reshape(B, -1)  # (B, D_latent=512)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.reshape(B, -1)  # (B, D_latent)

            NORM_METHOD = self.norm_method  # None | "std-scale" | "GN" | "BN"
            IGNORE_HIDDEN_STATES = False

            # Log.debug(f"income encoder_hidden_states: {encoder_hidden_states.min()}, {encoder_hidden_states.mean()}, {encoder_hidden_states.max()}")
            if NORM_METHOD is None:
                pass
            elif NORM_METHOD == "std-scale":
                factor = 0.5
                eps = 1e-9
                # scale the embedding
                emb_std = emb.std(dim=1, keepdim=True) + eps
                hid_std = encoder_hidden_states.std(dim=1, keepdim=True) + eps
                encoder_hidden_states *= (emb_std / hid_std) * factor
            elif NORM_METHOD == "GN":
                encoder_hidden_states = nn.functional.group_norm(encoder_hidden_states, 32)
                emb = nn.functional.group_norm(emb, 32)
            elif NORM_METHOD == "BN":
                running_mean = torch.zeros(encoder_hidden_states.shape[1]).to(encoder_hidden_states.device)
                running_var = torch.ones(encoder_hidden_states.shape[1]).to(encoder_hidden_states.device)
                encoder_hidden_states = nn.functional.batch_norm(encoder_hidden_states, running_mean, running_var)
                emb = nn.functional.batch_norm(emb, running_mean, running_var)
            else:
                raise NotImplementedError(f"Unknown norm method: {NORM_METHOD}")
            # Log.debug(f"final encoder_hidden_states: {encoder_hidden_states.min()}, {encoder_hidden_states.mean()}, {encoder_hidden_states.max()}")

            # no need for positional embedding in the input because we use convolution.
            if not IGNORE_HIDDEN_STATES:
                emb = emb + encoder_hidden_states  # (B, D_latent)

        # unet denoiser
        if self.version == "GMD_traj_release":  # to match the behavior of the original code
            sample[:, :, 196:224] = 0
            sample = self.unet(sample, cond=emb)  # (B, D_in, F)
            sample[:, :, 196:224] = 0
        elif self.version == "variable_and_8_divisible":
            assert F % 8 == 0
            sample = self.unet(sample, cond=emb)  # (B, D_in, F)
        else:
            raise NotImplementedError()

        return MdmUnetOutput(sample=sample)
