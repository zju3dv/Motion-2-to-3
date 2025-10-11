import torch
import torch.nn as nn
from torch.nn import functional as F
from .layers.hrnet.hrnet import get_hrnet


def flip(x):
    assert x.dim() == 3 or x.dim() == 4
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == "softmax":
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == "sampling":
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == "multiple_sampling":
        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError


class HRNetSMPLCam(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCam, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs["NUM_JOINTS"]
        self.norm_type = kwargs["POST"]["NORM_TYPE"]
        self.depth_dim = kwargs["EXTRA"]["DEPTH_DIM"]
        self.height_dim = kwargs["HEATMAP_SIZE"][0]
        self.width_dim = kwargs["HEATMAP_SIZE"][1]
        self.smpl_dtype = torch.float32
        self.pretrain_hrnet = kwargs["HR_PRETRAINED"]

        self.preact = get_hrnet(
            kwargs["HRNET_TYPE"],
            num_joints=self.num_joints,
            depth_dim=self.depth_dim,
            is_train=True,
            generate_feat=True,
            generate_hm=True,
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = (
            (1, 2),
            (4, 5),
            (7, 8),
            (10, 11),
            (13, 14),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
            (25, 26),
            (27, 28),
        )

        self.root_idx_smpl = 0

        init_cam = torch.tensor([0.9])
        self.register_buffer("init_cam", torch.Tensor(init_cam).float())

        self.decshape = nn.Linear(2048, 10)
        self.decphi = nn.Linear(2048, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, 29)

        self.focal_length = kwargs["FOCAL_LENGTH"]
        bbox_3d_shape = kwargs["BBOX_3D_SHAPE"] if "BBOX_3D_SHAPE" in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.input_size = 256.0

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]

        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]

        return heatmaps

    def forward(self, x, flip_test=False, **kwargs):
        B, c, h, w = x.shape
        assert h == w == int(self.input_size)
        # x0 = self.preact(x)
        out, x0 = self.preact(x)  # HRNet
        out = out.reshape(B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
        heatmaps = norm_heatmap(self.norm_type, out.reshape((B, self.num_joints, -1)))

        assert flip_test == False, "Seems to have a bug"
        if flip_test:
            flip_out, _ = self.preact(flip(x))
            flip_out = flip_out.reshape(B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            out_2 = self.flip_heatmap(flip_out)
            out_2 = out_2.reshape(B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            heatmaps_2 = norm_heatmap(self.norm_type, out_2.reshape((B, self.num_joints, -1)))
            heatmaps = (heatmaps + heatmaps_2) / 2

        assert heatmaps.dim() == 3, heatmaps.shape
        # maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x0 = heatmaps.sum((2, 3))  # (B, K, W)
        hm_y0 = heatmaps.sum((2, 4))  # (B, K, H)
        # hm_z0 = heatmaps.sum((3, 4))  # (B, K, D)

        hm_size = hm_x0.shape[-1]
        range_tensor = torch.arange(hm_size, dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)

        coord_x = hm_x0.matmul(range_tensor)
        coord_y = hm_y0.matmul(range_tensor)

        # convert to [0, 1]
        pred_joints24_xy = torch.cat([coord_x, coord_y], dim=2)[:, :24]  # (B, K, 2)
        pred_joints24_xy = pred_joints24_xy / (float(hm_size) - 1)

        # Get confidence for esach (x,y)
        # conf_hm = outmaps.sigmoid().max(2)[0][:, :24]
        conf_hm = heatmaps.sum(dim=2)[:, :24]  # (B, K, H, W)
        conf_hm = conf_hm.reshape(B * 24, 1, hm_size, hm_size)
        grid = pred_joints24_xy.reshape(B * 24, 1, 1, 2) * 2 - 1
        pred_c = F.grid_sample(conf_hm, grid[..., [1, 0]], mode="bilinear", align_corners=True).reshape(B, 24, 1)

        pred_kpts = torch.cat([pred_joints24_xy, pred_c], dim=2)
        return pred_kpts
