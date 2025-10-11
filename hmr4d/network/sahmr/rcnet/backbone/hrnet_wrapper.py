from pathlib import Path
import torch.nn as nn

from hmr4d.network.sahmr.sahmr.hrnet.config import config as hrnet_config
from hmr4d.network.sahmr.sahmr.hrnet.config import update_config as hrnet_update_config
from hmr4d.network.sahmr.sahmr.hrnet.hrnet_cls_net_featmaps import get_cls_net


class HRNetWrapper(nn.Module):
    def __init__(self, ver_name, dim_out):
        super().__init__()
        self.ver_name = ver_name
        if ver_name in ["metro_pretrained", "metro_wo_pretrained"]:
            hrnet_yaml = Path(__file__).parent / "cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            hrnet_update_config(hrnet_config, hrnet_yaml)
            if ver_name == "metro_pretrained":
                ckpt = "inputs/checkpoints/metro/metro_3dpw_state_dict.bin"
            else:
                ckpt = ""
            self.hrnet = get_cls_net(
                hrnet_config,
                pretrained=ckpt,
                prefix="backbone",
                featmap4s_only=True,
            )  # dim_out = 64
            # Disable gradient for unused parameters
            for k, v in self.hrnet.stage4[2].fuse_layers.named_parameters():
                if k[0] != "0":  # these parameters are not used
                    v.requires_grad = False
            self.out_conv = nn.Conv2d(64, dim_out, 1)
        else:
            raise KeyError

    def forward(self, x):
        hrnet_out = self.hrnet(x)
        out = self.out_conv(hrnet_out)
        return out
