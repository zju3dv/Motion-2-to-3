import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from . import BertConfig

from hmr4d.network.sahmr.sahmr.hrnet.config import config as hrnet_config
from hmr4d.network.sahmr.sahmr.hrnet.config import update_config as hrnet_update_config
from hmr4d.network.sahmr.sahmr.hrnet.hrnet_cls_net_featmaps import get_cls_net
from hmr4d.network.sahmr.sahmr.bert.sahmr_net_utils import CPOSE


m_cfg_default = OmegaConf.create(
    {
        "input_feat_dim": [2051, 512, 128],
        "hidden_feat_dim": [1024, 256, 128],
        "model_name_or_path": Path(__file__).parent / "bert-base-uncased/config.json",
        "num_hidden_layers": 4,
        "hidden_size": -1,
        "num_attention_heads": 4,
        "intermediate_size": -1,
    }
)


class METRO_enhanced(nn.Module):
    def __init__(self, m_cfg):
        super().__init__()
        m_cfg = OmegaConf.merge(m_cfg_default, m_cfg)

        # ---- Backbone
        hrnet_yaml = Path(__file__).parent.parent / "cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        self.backbone = get_cls_net(hrnet_config, pretrained="")  # this will be initialized with tr_encoder together

        # ---- Transformer Encoder
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
            update_params = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]

            for idx, param in enumerate(update_params):
                arg_param = getattr(m_cfg, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config, cpose_cfg=m_cfg)
            trans_encoder.append(model)

        self.trans_encoder = torch.nn.Sequential(*trans_encoder)
        self.conv_learn_tokens = torch.nn.Conv1d(49, 431 + 14, 1)
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)

    def load_from_pretrain(self, ckptname):
        if ckptname == "":
            return
        pretrained_dict = torch.load(ckptname, map_location="cpu")
        weight_to_load = {}
        for k in self.state_dict():
            if k in pretrained_dict:
                weight_to_load[k] = pretrained_dict[k]
        missing_keys, unexpected_keys = self.load_state_dict(weight_to_load, strict=False)
        missing_keys = [k for k in missing_keys if not ("loftr" in k or "linear_cp" in k)]
        missing_keys = [k for k in missing_keys if not ("v4" in k)]
        assert len(missing_keys) == 0

    def forward(self, batch):
        images = batch["imgs_net_input"]
        init_joints = batch["init_joints14"]
        init_verts = batch["init_verts431"]
        B, J = init_joints.shape[:2]

        # how to use scene pointcloud
        pred_coord = batch["flag_pred_coord"]  # crm
        corr_pc_type = batch["flag_corr_pc_type"]  # pcSeg

        corr_points = batch[f"pred_{pred_coord}_pcSeg"]
        corr_mask = batch["pcSeg_mask"]
        corr_pids = batch["pcSeg_pids"]
        pid_to_verts431 = batch["pid_to_verts431"]
        cp_xyz = None

        # extract image feature maps using a CNN backbone, (B, 2048, 7, 7). where 7 = 224/32
        # use linear map 49 to 14+431. (B, 445, 2048)
        # img_fmap = batch['feat_global']  # (B, 2048, 7, 7)
        img_fmap = self.backbone(images)  # (B, 2048, 7, 7)
        img_tokens = self.conv_learn_tokens(img_fmap.view(B, 2048, -1).transpose(1, 2))

        # concatinate image feat and template mesh
        ref_vertices = torch.cat([init_joints, init_verts], dim=1)  # (B, 445, 3)
        features = torch.cat([ref_vertices, img_tokens], dim=2)  # (B, 445, 3+2048)

        # concatinate image feat and corr points
        vert431_token = img_tokens[:, J:]  # (B, 431, 2048)
        part_token = torch.stack(
            [vert431_token[:, pid_to_verts431[i]].mean(1) for i in range(1, 8)], dim=1  # avg pooling
        )  # (B, 7, 2048)
        part_token = F.pad(part_token, (0, 0, 1, 0), value=0.0)  # (B, 1+7, 2048)
        expand_part_token = torch.stack([part_token[b, corr_pids[b]] for b in range(B)])  # (B, 500, 2048)
        features_cp = torch.cat([corr_points, expand_part_token], dim=2)  # (B, 500, 3+2048)

        # forward pass
        for i in range(len(self.trans_encoder)):
            features, features_cp = self.trans_encoder[i](
                features, features_cp, corr_mask, corr_pids, pid_to_verts431, cp_xyz  # main
            )  # optional
            batch[f"cpose_img_feat_{i}"] = features
            batch[f"cpose_cp_feat_{i}"] = features_cp

        pred_vertices_sub2 = features[:, J:, :]  # pred_3d_joints = features[:, :J, :]

        temp_transpose = pred_vertices_sub2.transpose(1, 2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)
        pred_vertices_full = pred_vertices_full.transpose(1, 2)

        # returns
        batch["pred_verts"] = pred_vertices_full
        batch["pred_corr_pc"] = features_cp
