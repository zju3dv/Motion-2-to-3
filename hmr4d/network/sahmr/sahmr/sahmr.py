import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr4d.network.sahmr.sahmr.bert.modeling_metro_enhanced import METRO_enhanced
from hmr4d.network.sahmr.rcnet.rc_net import RCNet, get_pc_belong_to_vox_seg


class SAHMR(nn.Module):
    def __init__(self, rc_cfg, m_cfg, rc_ckpt="", m_ckpt=""):
        super().__init__()

        # ===== Networks ===== #
        # module 1
        self.rcnet = RCNet(**rc_cfg).requires_grad_(False).eval()
        self.load_rcnet_weights(rc_ckpt)

        # module 2
        self.metro_enhanced = METRO_enhanced(m_cfg)
        self.metro_enhanced.load_from_pretrain(m_ckpt)

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1), False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1), False)

    def load_rcnet_weights(self, ckpt):
        if ckpt == "":
            return
        pretrained_weights = torch.load(ckpt, "cpu")["network"]
        pretrained_weights = {k[4:]: v for k, v in pretrained_weights.items()}  # remove 'net.'
        self.rcnet.load_state_dict(pretrained_weights)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module._get_name() == "RCNet":
                continue
            module.train(mode)
        return self

    def forward_mainstep(self, batch, pred_c_pelvis_refined):
        # # ===== Handle input ===== #
        batch["imgs_net_input"] = (batch["image"] - self.img_mean) / self.img_std  # (B, 3, 224, 224)
        batch["flag_pred_coord"] = "crm"
        batch["flag_corr_pc_type"] = "pcSeg"
        smpl = batch["smpl"]
        self.metro_enhanced(batch)

        # ====== Handle output ===== #
        pred_verts = batch["pred_verts"]
        batch["pred_crm_pcSeg_after"] = batch["pred_corr_pc"]

        # handle pred joints and verts
        batch["pred_cr_verts"], batch["pred_cr_joints14"] = smpl.get_r_h36m_verts_joints14(pred_verts)  # MPJPE good

        batch["pred_c_verts"] = pred_verts + pred_c_pelvis_refined
        with torch.no_grad():
            batch["pred_c_joints14"] = smpl.get_h36m_joints14(batch["pred_c_verts"])

    def forward(self, batch):
        # ===== RCNet ===== #
        with torch.no_grad():
            self.rcnet(batch)
        # prepare contact points
        get_pc_belong_to_vox_seg(batch)
        pred_c_pcSeg, pred_pcSeg_pids, pred_pcSeg_mask = prepare_N_pcSeg(batch, 500)
        batch["pred_c_pcSeg"] = pred_c_pcSeg  # (B, 500 ,3)
        batch["pcSeg_pids"] = pred_pcSeg_pids  # (B, 500)
        batch["pcSeg_mask"] = pred_pcSeg_mask  # (B, 500)

        # prepare cam-regularize-metro_enhanced coord
        pred_c_pelvis_refined = batch["pred_c_pelvis_refined"].detach().clone()
        if self.training:
            # 1. replace wrong wrong c_pelvis prediction with gt
            replace_mask = ((pred_c_pelvis_refined - batch["gt_c_pelvis"]).norm(p=2, dim=-1) > 0.5).squeeze(1)
            noisy_gt_c_pelvis = batch["gt_c_pelvis"].clone()
            noisy_gt_c_pelvis += 0.08 * torch.randn_like(noisy_gt_c_pelvis)  # (B, 1, 3)
            pred_c_pelvis_refined[replace_mask] = noisy_gt_c_pelvis[replace_mask]

        batch["pred_crm_pcSeg"] = batch["pred_c_pcSeg"] - pred_c_pelvis_refined

        # 2. enhanced metro
        self.forward_mainstep(batch, pred_c_pelvis_refined)


@torch.no_grad()
def prepare_N_pcSeg(batch, N=500):
    """
    Returns:
        pred_c_pcSeg: (B, N, 3)
        pred_pcSeg_pids: (B, N)
        pred_pcSeg_mask: (B, N)
    """
    B = batch["image"].shape[0]
    pred_c_pcSeg_all, pred_pcSeg_all_pids = batch["pred_c_pcSeg_all"], batch["pred_pcSeg_all_pids"]
    pred_c_pcSeg, pred_pcSeg_pids = [], []
    for b in range(B):
        n = len(pred_c_pcSeg_all[b])  # >=1
        if n <= N:
            pred_c_pcSeg.append(F.pad(pred_c_pcSeg_all[b], (0, 0, 0, N - n), value=0.0))
            pred_pcSeg_pids.append(F.pad(pred_pcSeg_all_pids[b], (0, N - n), value=0))
        else:
            idx = torch.randint(n, (N,))
            pred_c_pcSeg.append(pred_c_pcSeg_all[b][idx])
            pred_pcSeg_pids.append(pred_pcSeg_all_pids[b][idx])
    pred_c_pcSeg = torch.stack(pred_c_pcSeg)
    pred_pcSeg_pids = torch.stack(pred_pcSeg_pids)
    pred_pcSeg_mask = pred_pcSeg_pids != 0

    return pred_c_pcSeg, pred_pcSeg_pids, pred_pcSeg_mask
