import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.geo_transform import kabsch_algorithm_batch, similarity_transform_batch
from hmr4d.utils.wis3d_utils import add_motion_as_lines
from pytorch3d.transforms import axis_angle_to_matrix
from .optim_rotation import R_from_wy


@torch.inference_mode(mode=False)
def update_pred_with_wst(pred, w, s, t, max_diff_s=0.1):
    """
    scale = 1 + 2 * (sigmoid(s) - 0.5) * max_diff_s
    """
    # Rotation matrix along y axis. because we trust our 3D prior for gravity direction
    R = R_from_wy(w)  # (B, L, 3, 3)

    # Use local coordinate, and we do not touch y-value. because we trust our 3D prior
    offset_x0z = pred[..., [0], :].clone()  # (B, L, 1, 3)
    offset_x0z[..., 1] = 0

    # pred_update (B, L, J, 3)
    pred_update = einsum(pred - offset_x0z, R, "b l n c, b l d c -> b l n d")  # rotate
    s = 1 + 2 * (torch.sigmoid(s) - 0.5) * max_diff_s  # limit the scale
    if len(s.shape) == 2:  # one s for one frame
        pred_update = pred_update * s[:, :, None, None]  # scale
    else:  # one s for one video
        pred_update = pred_update * s[:, None, None, None]  # scale
    pred_update = pred_update + t + offset_x0z  # translate
    return pred_update


@torch.inference_mode(mode=False)
def find_wst_transform(
    pred, T_ayfz2c, obs_c_p2d, length, obs_cr_p3d=None, lr=0.01, max_iter=50, verbose=False, wis3d=None
):
    """(B, L, J, 3), (B, 1/L, 4, 4), (B, L, J, 2), (B)"""
    assert obs_cr_p3d == None
    B, max_L, J = pred.shape[:3]
    device = pred.device
    assert len(pred.shape) == len(T_ayfz2c.shape) == len(obs_c_p2d.shape) == 4

    # parameters to optimize
    w = torch.zeros(B, max_L, device=device).requires_grad_()
    s = torch.ones(B, device=device).requires_grad_()

    InitTxz = True
    if InitTxz:
        c_root2d = obs_c_p2d[:, :, 0]
        B, L, _ = c_root2d.shape

        # line_point + line_dir * x = point
        # c-coordinate
        c_ld = F.pad(c_root2d, (0, 1), value=1.0)  # (B, L, 3)
        c_lp = torch.zeros_like(c_ld)

        # w-coordinate
        assert len(T_ayfz2c.shape) == 4
        T_c2w = torch.inverse(T_ayfz2c)  # (B, L, 4, 4)
        R_c2w = T_c2w[:, :, :3, :3]  # (B, L, 3, 3)
        t_c2w = T_c2w[:, :, :3, 3]  # (B, L, 3)
        w_lp = einsum(R_c2w, c_lp, "b l c d, b l d -> b l c") + t_c2w  # (B, L, 3)
        w_ld = einsum(R_c2w, c_ld, "b l c d, b l d -> b l c")

        # solve y=pred_root_y
        w_lend_y0 = pred[:, :, 0, 1]  # (B, L)
        x = (w_lend_y0 - w_lp[:, :, 1]) / w_ld[:, :, 1]  # (B, L)
        w_lend = w_lp + x.unsqueeze(-1) * w_ld  # (B, L, 3)
        tx = w_lend[:, :, 0]  # (B, L)
        tz = w_lend[:, :, 2]  # (B, L)

        t_init_target = torch.stack([tx, w_lend_y0, tz], -1).clone().detach()  # (B, L, 3)
        t_init_target_c = apply_T_on_points(t_init_target[:, :, None], T_ayfz2c)  # (B, L, 1, 3)
        z_mask = t_init_target_c[..., 2] < 0
        t_init_target_c[z_mask] = -1 * t_init_target_c[z_mask]
        t_init_target = apply_T_on_points(t_init_target_c, torch.inverse(T_ayfz2c))[:, :, 0]  # (B, L, 3)
        tx = t_init_target[:, :, 0]
        tz = t_init_target[:, :, 2]

        tx = (tx.clone().detach() - pred[:, :, 0, 0]).requires_grad_()
        ty = torch.zeros(B, max_L, device=device).detach()
        tz = (tz.clone().detach() - pred[:, :, 0, 2]).requires_grad_()
    else:
        tx = torch.zeros(B, max_L, device=device).requires_grad_()
        ty = torch.zeros(B, max_L, device=device).detach()
        tz = torch.zeros(B, max_L, device=device).requires_grad_()

    optimizer = torch.optim.Adam([w, s, tx, tz], lr=lr)

    # # Add another
    # s_cr = torch.ones(B, device=device).requires_grad_()
    # optimizer.add_param_group({"params": s_cr})

    length_mask = torch.arange(max_L, device=pred.device) < length.reshape(B, 1)
    scale_cr2pred = None
    # pred_anchor = pred.detach().clone()  # (B, J, 3)
    for i in range(max_iter):
        loss = 0
        t = torch.stack([tx, ty, tz], -1)[:, :, None, :]  # (B, L, 1, 3)
        pred_update = update_pred_with_wst(pred, w, s, t)  # (B, L, J, 3)
        loss += (pred_update[:, 1:] - pred_update[:, :-1]).pow(2).sum((-1, -2))[length_mask[:, 1:]].mean()

        if wis3d is not None:
            if i % 10 == 0 or i == max_iter - 1:
                add_motion_as_lines(pred_update[0], wis3d, name=f"during_optimize_{i}")

        # transform to cam-coordinate
        c_pred_update = apply_T_on_points(pred_update, T_ayfz2c.clone())  # (B, L, J, 3)
        # make sure the last dim of c_pred_update is not zero
        if (c_pred_update[..., 2:].abs() < 1e-4).sum() > 0:
            print("Warning !!!!")
            breakpoint()
        c_pred2d_update = c_pred_update[..., :2] / (c_pred_update[..., 2:] + 1e-4)  # (B, L, J, 2)
        if torch.isinf(c_pred2d_update).any():
            print("Warning !!!!")
            breakpoint()

        LOSS_ON_SELECTED_JOINTS = False
        if LOSS_ON_SELECTED_JOINTS:
            loss += (c_pred2d_update - obs_c_p2d).pow(2)[:, :, [0, 7, 8, 20, 21]].sum((-1, -2))[length_mask].mean()
        else:  # Full joints
            loss += (c_pred2d_update - obs_c_p2d).pow(2).sum((-1, -2))[length_mask].mean()  # (B, L)

        # use p3d as regularization
        # if obs_cr_p3d is not None:
        #     cr_pred_update = c_pred_update - c_pred_update[..., :1, :]
        #     # loss_crp3d = (obs_cr_p3d.clone() * s_cr[:, None, None, None] - cr_pred_update).pow(2).sum((-1, -2))
        #     if scale_cr2pred is None:
        #         with torch.no_grad():
        #             (scale_cr2pred, _), _ = similarity_transform_batch(obs_cr_p3d, cr_pred_update)
        #             # scale_cr2pred = scale_cr2pred.mean(1, keepdim=True)  # even worse
        #     loss_crp3d = (obs_cr_p3d.clone() * scale_cr2pred - cr_pred_update).pow(2).sum((-1, -2))
        #     # loss_crp3d = (obs_cr_p3d.clone() - cr_pred_update).pow(2).sum((-1, -2))
        #     loss += loss_crp3d[length_mask].mean()

        # Add some regularization
        # s should be close to 1
        # loss += (s - 1).pow(2).mean()

        # w and t should not change too much
        # loss += (w[:, 1:] - w[:, :-1]).pow(2)[length_mask[:, 1:]].mean()
        # loss += (tx[:, 1:] - tx[:, :-1]).pow(2)[length_mask[:, 1:]].mean()
        # loss += (tz[:, 1:] - tz[:, :-1]).pow(2)[length_mask[:, 1:]].mean()

        # loss += (ty - 0).pow(2)[length_mask].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            print(f"{i} loss: {loss.item()}")

    t = torch.stack([tx, torch.zeros_like(tx), tz], -1)[:, :, None, :]  # (B, L, 1, 3)
    # t = torch.stack([tx, ty, tz], -1)[:, :, None, :]  # (B, L, 1, 3)
    return w.detach(), s.detach(), t.detach(), t_init_target
