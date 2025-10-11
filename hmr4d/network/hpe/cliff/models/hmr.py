import torch
import torch.nn as nn

from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from .head.hmr_head_cliff import HMRHeadCLIFF
from .head.smplx_cam_head import SMPLXCamHead
from .head.smplx_cam_head_proj import SMPLXCamHeadProj
from ..core.config import PRETRAINED_CKPT_FOLDER


class HMR(nn.Module):
    def __init__(self, backbone="hrnet_w48-conv", img_res=224, focal_length=5000, pretrained_ckpt=None, hparams=None):
        super(HMR, self).__init__()
        self.hparams = hparams

        # Initialize backbone
        backbone, use_conv = backbone.split("-")
        pretrained_ckpt = backbone + "-" + pretrained_ckpt
        pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
        self.backbone = eval(backbone)(
            pretrained_ckpt_path=pretrained_ckpt_path,
            downsample=True,
            use_conv=(use_conv == "conv"),
        )
        # Initialize head (smplx, beta11 used in BEDLAM)
        self.head = HMRHeadCLIFF(
            num_input_features=get_backbone_info(backbone)["n_output_channels"],
            backbone=backbone,
        )
        if hparams.DATASET.proj_verts:  # train with projected downsampled verts432
            self.smpl = SMPLXCamHeadProj(img_res=img_res)
        else:  # inference default
            self.smpl = SMPLXCamHead(img_res=img_res)

    def forward(self, images, bbox_scale=None, bbox_center=None, img_w=None, img_h=None, fl=None):
        batch_size = images.shape[0]

        if fl is not None:
            # GT focal length
            focal_length = fl
        else:
            # Estimate focal length
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialze cam intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w / 2.0
        cam_intrinsics[:, 1, 2] = img_h / 2.0

        # Taken from CLIFF repository
        cx, cy = bbox_center[:, 0], bbox_center[:, 1]
        b = bbox_scale * 200
        bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)  # [-1, 1]
        bbox_info[:, 2] = bbox_info[:, 2] / cam_intrinsics[:, 0, 0]  # [-1, 1]
        bbox_info = bbox_info.cuda().float()
        features = self.backbone(images.cuda())
        hmr_output = self.head(features, bbox_info=bbox_info)

        # Assuming prediction are in camera coordinate
        smpl_output = self.smpl(
            rotmat=hmr_output["pred_pose"],
            shape=hmr_output["pred_shape"],
            cam=hmr_output["pred_cam"],
            cam_intrinsics=cam_intrinsics,
            bbox_scale=bbox_scale,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            normalize_joints2d=False,
        )

        smpl_output.update(hmr_output)
        smpl_output.update({"cam_intrinsics": cam_intrinsics})
        return smpl_output

    def forward_minimal(self, images, bbox_info):
        features = self.backbone(images)
        hmr_output = self.head(features, bbox_info=bbox_info)
        return hmr_output
