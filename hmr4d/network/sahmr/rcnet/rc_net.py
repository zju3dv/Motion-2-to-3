import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr4d.network.sahmr.rcnet.backbone.hrnet_wrapper import HRNetWrapper
from hmr4d.network.sahmr.rcnet.backbone.spvcnn import SPVCNN

import torchsparse.nn.functional as SF
from torchsparse import SparseTensor, PointTensor
from hmr4d.utils.torchsparse_utils import initial_voxelize
from hmr4d.utils.geo_transform import (
    cvt_from_bi01_p2d,
    cvt_to_bi01_p2d,
    project_p2d,
    unproject_p2d,
    l2uv_index,
    get_nearby_points,
)


class RCNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.voxel_size = 0.05
        self.d_an2an = 0.5
        self.r_keep = 1.25
        self.train_anchor_err_thr = 0.5

        # ===== backbone ===== #
        # Bbx-Image
        dim_root = 2
        self.dim_bi4s_seg = 7  # auxillary segmentation task
        ver_name = kwargs.get("hrnet_ver_name", "metro_pretrained")
        self.root_net = HRNetWrapper(ver_name, dim_out=dim_root + 32 + self.dim_bi4s_seg)

        # Scene-Pointcloud
        dim_coord = 3
        self.dim_refine_xyz = dim_refine_xyz = 1 + dim_coord  # confidence + xyz
        self.dim_voxSeg = 8  # segmentation
        self.contact_net = SPVCNN(d_in=dim_coord + 32, num_classes=dim_refine_xyz + self.dim_voxSeg)

        # ===== Image ===== #
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1), False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1), False)

    def forward(self, batch):
        # ===== Handle input ===== #
        imgs_net_input = (batch["image"] - self.img_mean) / self.img_std  # (B, 3, 224, 224)
        B = imgs_net_input.size(0)

        # ===== RootNet process ===== #
        if self.training and batch.get("flag_train_extra_flip", False):
            assert self.dim_bi4s_seg != 0  # for seg only
            root_net_out = self.root_net(torch.cat([imgs_net_input, torch.flip(imgs_net_input, [-1])], dim=0))  # 2B
            batch["pred_bi4s_seg_hm"] = root_net_out[:B, 34:, :, :]
            batch["pred_bi4s_seg_hm_flip"] = root_net_out[B:, 34:, :, :]
            root_net_out = root_net_out[:B]
        else:
            root_net_out = self.root_net(imgs_net_input)
            batch["pred_bi4s_seg_hm"] = root_net_out[:, 34:, :, :]  # (B, 7, H, W)
        parse_pelvis_hm_output(root_net_out[:, :2, :, :], batch)

        # ===== Handle intermediate results ===== #
        fmap_img = root_net_out[:, 2:34, :, :]

        with torch.no_grad():
            # 1. use predicted pelvis2.5D to get nearby scene pointclouds
            # in training, we use noisy(on z-axis) gt to replace bad prediction(>1m)
            c_root_anchors = batch["pred_c_pelvis"].clone().detach()  # (B, 1, 3)
            if self.training:
                gt_c_pelvis = batch["gt_c_pelvis"]  # (B, 1, 3)
                noisy_gt_c_pelvis = gt_c_pelvis.clone()
                noisy_gt_c_pelvis += 0.5 * torch.randn_like(noisy_gt_c_pelvis)  # (B, 1, 3)
                training_noisy_gt_root_mask = (
                    (c_root_anchors - gt_c_pelvis).norm(p=2, dim=-1) > self.train_anchor_err_thr
                ).squeeze(1)
                batch["training_noisy_gt_root_mask"] = training_noisy_gt_root_mask
                c_root_anchors[training_noisy_gt_root_mask] = noisy_gt_c_pelvis[training_noisy_gt_root_mask]

            # +add 2 more anchors along z-axis to improve recall
            c_root_anchors = c_root_anchors.repeat(1, 3, 1)
            c_root_anchors[:, 0, 2] -= self.d_an2an
            c_root_anchors[:, 2, 2] += self.d_an2an
            c_pcFst_nearby = [
                get_nearby_points(batch["c_pcFst_all"][b], c_root_anchors[b], padding=self.r_keep, p=2)
                for b in range(B)
            ]
            for b in range(B):
                if len(c_pcFst_nearby[b]) == 0:  # no nearby points, use all points
                    c_pcFst_nearby[b] = batch["c_pcFst_all"][b]
            batch["c_pcFst_nearby"] = c_pcFst_nearby

            # 2. voxelize these points
            # pc_xyzb = torch.cat([F.pad(p, (0, 1), value=b) for b, p in enumerate(batch['c_pcFst_all'])], dim=0)
            pc_xyzb = torch.cat([F.pad(p, (0, 1), value=b) for b, p in enumerate(c_pcFst_nearby)], dim=0)
            z = PointTensor(coords=pc_xyzb, feats=pc_xyzb[:, :3])  # Raw points
            x = initial_voxelize(z, 1, self.voxel_size)  # Voxelize, 归并[0, voxel_size)的点, 注意z的坐标被voxel_size改变
            x_ori_xyz = x.C[:, :3] * self.voxel_size + self.voxel_size / 2  # 还原到原本的分辨率
            batch["voxel_size"] = self.voxel_size

        # 3. unproject image feature to voxel
        v_vox_coord, c_vox_coord, feat = [], [], []
        for b in range(B):
            b_mask = x.C[:, -1] == b
            v_vox_coord.append(x.C[b_mask])  # (N, 4), in voxel coord
            c_vox_coord.append(x_ori_xyz[b_mask])  # (N, 3) in cam coords

            bi01_p2d = cvt_to_bi01_p2d(
                project_p2d(c_vox_coord[-1][None], batch["K"][[b]]), batch["bbx_lurb"][[b]]
            )  # (1, N, 2)
            feat_unproj = F.grid_sample(fmap_img[[b]], bi01_p2d[:, None] * 2 - 1, align_corners=True)[0, :, 0].permute(
                1, 0
            )
            feat_coord = c_root_anchors[b, 1] - c_vox_coord[-1]  # zero-center
            feat.append(torch.cat([feat_coord, feat_unproj], dim=1))  # (N, 3+F)
        voxel_feats = SparseTensor(torch.cat(feat, dim=0), torch.cat(v_vox_coord, dim=0), x.s)  # (S, 3+F)
        # batch['c_voxFst_all'] = c_vox_coord
        batch["c_voxFst_nearby"] = c_vox_coord

        # ===== ContactNet process ===== #
        voxel_out = self.contact_net(voxel_feats)  # (S, 1+8)

        pred_vox_xyz_conf = [voxel_out.F[b == voxel_out.C[:, -1], :1] for b in range(B)]
        pred_vox_xyz_offset = [voxel_out.F[b == voxel_out.C[:, -1], 1 : self.dim_refine_xyz] for b in range(B)]
        pred_vox_seg = [voxel_out.F[b == voxel_out.C[:, -1], self.dim_refine_xyz :] for b in range(B)]

        batch["pred_vox_xyz_conf"] = pred_vox_xyz_conf
        batch["pred_vox_xyz_offset"] = pred_vox_xyz_offset
        batch["pred_vox_seg"] = pred_vox_seg
        batch["pred_vox_segid"] = [v.max(1)[1] for v in pred_vox_seg]

        # compute refined 3d
        batch["pred_c_pelvis_refined"] = c_root_anchors[:, [1], :]
        for b in range(B):
            conf, offset = pred_vox_xyz_conf[b], pred_vox_xyz_offset[b]
            xyz = (torch.softmax(conf.sigmoid(), 0) * (c_vox_coord[b] + offset)).sum(0)
            if not torch.isnan(xyz).any():
                batch["pred_c_pelvis_refined"][b, 0] = xyz


def parse_pelvis_hm_output(root_net_out, batch):
    pred_bi4s_pelvis_hm, pred_bi4s_rdepth_hm = root_net_out[:, [0]], root_net_out[:, [1]]
    batch["pred_bi4s_pelvis_hm"] = pred_bi4s_pelvis_hm  # (B, 1, H/4, W/4)
    batch["pred_bi4s_rdepth_hm"] = pred_bi4s_rdepth_hm  # (B, 1, H/4, W/4)

    # 1. root xy (easy to learn)
    with torch.no_grad():
        B, _, _, W = pred_bi4s_pelvis_hm.shape
        l = pred_bi4s_pelvis_hm.view(B, -1).argmax(1)
        uv = l2uv_index(l, W)
        batch["pred_bi4s_pelvis_2d"] = uv  # (B, 2)
        batch["pred_bi01_pelvis_2d"] = uv / (W - 1)  # (B, 2)

    # 2. root depth (with ambiguity)
    with torch.no_grad():
        # pelvis zw1f to z
        pred_zw1f = torch.stack([pred_bi4s_rdepth_hm[b, 0, uv[b, 1], uv[b, 0]] for b in range(B)])
        pred_z = pred_zw1f * batch["bi_f"] / batch["bi_w"]
        batch["pred_c_pelvis_z"] = pred_z  # (B,)

    # 3. root xyz in cam-coord
    with torch.no_grad():
        pred_ri_pelvis_2d = cvt_from_bi01_p2d(batch["pred_bi01_pelvis_2d"][:, None], batch["bbx_lurb"])
        pred_c_pelvis = unproject_p2d(pred_ri_pelvis_2d, pred_z.reshape(-1, 1, 1), batch["K"])
        batch["pred_c_pelvis"] = pred_c_pelvis  # (B, 1, 3)


def get_pc_belong_to_vox_seg(batch):
    B = batch["image"].shape[0]
    pred_c_pcSeg_all = []
    pred_pcSeg_all_pids = []
    for b in range(B):
        mask_vox_seg = batch["pred_vox_segid"][b] != 0  # sum to N
        if mask_vox_seg.sum() == 0:
            pred_c_pcSeg_all.append(batch["c_voxFst_nearby"][b][[0]])
            pred_pcSeg_all_pids.append(batch["pred_vox_segid"][b][[0]])
            continue

        v_vox_seg_C = F.pad((batch["c_voxFst_nearby"][b] / batch["voxel_size"]).floor(), (0, 1), value=b).int()[
            mask_vox_seg
        ]
        vox_hash = SF.sphash(v_vox_seg_C)
        v_pc_C = F.pad((batch["c_pcFst_nearby"][b] / batch["voxel_size"]).floor(), (0, 1), value=b).int()
        pc_hash = SF.sphash(v_pc_C)
        idx_query = SF.sphashquery(pc_hash, vox_hash)
        # add to all list
        pred_c_pcSeg_all.append(batch["c_pcFst_nearby"][b][idx_query != -1])
        pred_pcSeg_all_pids.append(batch["pred_vox_segid"][b][mask_vox_seg][idx_query[idx_query != -1]])

    batch["pred_c_pcSeg_all"] = pred_c_pcSeg_all
    batch["pred_pcSeg_all_pids"] = pred_pcSeg_all_pids
