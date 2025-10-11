import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.utils.geo_transform import apply_T_on_points

from .hrnet.config import config as hrnet_config
from .hrnet.config import update_config as hrnet_update_config
from .hrnet.hrnet_cls_net_featmaps import get_cls_net

from .bert import BertConfig, CPOSE
from .bert import CPOSE_Body_Network as CPOSE_Network


class CorrPose(nn.Module):
    def __init__(self, m_cfg, cfg):
        super().__init__()
        self.m_cfg = m_cfg
        self.cfg = cfg

        # ===== backbone ===== #
        hrnet_yaml = 'src/networks/pose/metro/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained="")

        # ===== transformer ===== #
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        input_feat_dim = m_cfg.input_feat_dim
        hidden_feat_dim = m_cfg.hidden_feat_dim
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, CPOSE
            config = config_class.from_pretrained(m_cfg.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            m_cfg.hidden_size = hidden_feat_dim[i]
            m_cfg.intermediate_size = -1

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(m_cfg, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config, cpose_cfg=m_cfg)
            trans_encoder.append(model)
        trans_encoder = torch.nn.Sequential(*trans_encoder)

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        self.learn_cam_params = m_cfg.learn_cam_params
        self.pred_coord = m_cfg.pred_coord  # cr, c, cray
        self.corr_pc_type = m_cfg.corr_pc_type # points431, pcSeg, pcSeg_wocorr
        self.cpose_network = CPOSE_Network(backbone, trans_encoder, 
                                           self.learn_cam_params, self.corr_pc_type)

        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1), False)
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1), False)

    def forward(self, batch):
        # ===== Handle input ===== #
        smpl = batch['smpl']
        batch['imgs_net_input'] = (batch['image'] - self.img_mean) / self.img_std  # (B, 3, 224, 224)
        batch['flag_pred_coord'] = self.pred_coord
        batch['flag_corr_pc_type'] = self.corr_pc_type

        # ===== Network process ===== #
        self.cpose_network(batch)

        # ====== Handle output ===== #
        pred_verts = batch['pred_verts']
        pred_cam = batch['pred_cam']
        pred_corr_pc = batch['pred_corr_pc']

        # handle pred joints and verts
        if self.pred_coord == 'cr':
            pred_cr_verts, pred_cr_joints14 = smpl.get_r_h36m_verts_joints14(pred_verts)
            pred_cr_corr_pc = pred_corr_pc            
        elif self.pred_coord == 'cray':
            pred_cray_verts, pred_cray_joints14 = smpl.get_r_h36m_verts_joints14(pred_verts)
            batch['pred_cray_verts'] = pred_cray_verts
            batch['pred_cray_joints14'] = pred_cray_joints14
            batch['pred_cray_corr_pc'] = pred_corr_pc
            pred_cr_verts = apply_T_on_points(pred_cray_verts, batch['T_cray2cr'])
            pred_cr_joints14 = apply_T_on_points(pred_cray_joints14, batch['T_cray2cr'])
            pred_cr_corr_pc = apply_T_on_points(pred_corr_pc, batch['T_cray2cr'])

        batch['pred_cr_verts'] = pred_cr_verts  # (B, 6890, 3)
        batch['pred_cr_joints14'] = pred_cr_joints14  # (B, 14, 3)
        batch['pred_cr_corr_pc'] = pred_cr_corr_pc  # (B, 431, 3)

        # from_verts, project with pred_cam
        if self.learn_cam_params:
            pred_bi_joints14_2d = orthogonal_proj_points(pred_cr_joints14, pred_cam, batch['image'].size(-1))
            batch.update({'pred_cam': pred_cam, 'pred_bi_joints14_2d': pred_bi_joints14_2d})


def orthogonal_proj_points(X, camera, W):
    """Perform orthogonal projection of 3D points X using the camera parameters and image size
    Assume figure range is [-1, 1]
    Args:
        X: [B, N, 3]
        camera: [B, 3]
    Returns:
        Projected 2D points: [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)

    X_2d = X_2d * W / 2 + W / 2
    return X_2d
