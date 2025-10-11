import torch
from omegaconf import OmegaConf
from .HRNetWithCam import HRNetSMPLCam
from pathlib import Path

MEAN = torch.tensor([0.406, 0.457, 0.480]).view(3, 1, 1)
STD = torch.tensor([0.225, 0.224, 0.229]).view(3, 1, 1)
input_size = 256  # (256, 256)


cfg = OmegaConf.load(f"{Path(__file__).parent}/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml")
cfg_model = cfg.MODEL.copy()
cfg_model.pop("TYPE")
