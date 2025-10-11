import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from .vit import vit
from .smpl_head import SMPLTransformerDecoderHead


class HMR2(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = vit(cfg)
        self.smpl_head = SMPLTransformerDecoderHead(cfg)

    def forward(self, batch):
        # Use RGB image as input
        x = batch["img"]
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        token_out = self.smpl_head(conditioning_feats)  # (B, 1024)
        return token_out

        # # Store useful regression outputs to the output dict
        # output = {}
        # output['pred_cam'] = pred_cam
        # output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # # Compute camera translation
        # device = pred_smpl_params['body_pose'].device
        # dtype = pred_smpl_params['body_pose'].dtype
        # focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        # pred_cam_t = torch.stack([pred_cam[:, 1],
        #                           pred_cam[:, 2],
        #                           2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        # output['pred_cam_t'] = pred_cam_t
        # output['focal_length'] = focal_length

        # # Compute model vertices, joints and the projected joints
        # pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        # pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        # pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        # smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        # pred_keypoints_3d = smpl_output.joints
        # pred_vertices = smpl_output.vertices
        # output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        # output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        # pred_cam_t = pred_cam_t.reshape(-1, 3)
        # focal_length = focal_length.reshape(-1, 2)
        # pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
        #                                            translation=pred_cam_t,
        #                                            focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        # output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        # return output
