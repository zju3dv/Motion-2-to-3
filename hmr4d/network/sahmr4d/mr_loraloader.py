import torch
import torch.nn as nn
from hmr4d.network.base_arch.lora import LoRALinearLayer, LoRAConv1dLayer, LoRACompatibleConv1d, LoRACompatibleLinear
from hmr4d.utils.pylogger import Log


class LoraLoader:
    def create_lora_weights(self):
        targets = self.args_lora.targets
        if "unet" in targets:
            for k, m in list(self.unet.named_modules()):
                is_linear = isinstance(m, LoRACompatibleLinear)
                is_conv = isinstance(m, LoRACompatibleConv1d)
                is_conv1x1 = is_conv and m.kernel_size == (1,)

                examine_type = isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear)
                if examine_type and (not (is_linear or is_conv)):
                    Log.warn(f"Warning: {k} is not lora compatible, skipping...")
                    continue

                if is_linear:
                    lora = LoRALinearLayer(
                        in_features=m.in_features,
                        out_features=m.out_features,
                        rank=self.args_lora.rank,
                        network_alpha=self.args_lora.network_alpha,
                    )
                    m.set_lora_layer(lora)
                elif is_conv:
                    lora = LoRAConv1dLayer(
                        in_features=m.in_channels,
                        out_features=m.out_channels,
                        rank=self.args_lora.rank,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        network_alpha=self.args_lora.network_alpha,
                    )
                    m.set_lora_layer(lora)
        else:
            raise NotImplementedError

    # def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
    #     state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
    #     self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)
    #     self.load_lora_into_text_encoder(
    #         state_dict,
    #         network_alphas=network_alphas,
    #         text_encoder=self.text_encoder,
    #         lora_scale=self.lora_scale,
    #     )

    def save_lora_weights(self, save_path):
        raise NotImplementedError  # TODO: save lora in a separate file.
        # currently the whole checkpoint is saved, and the model can not be loaded correctly.
