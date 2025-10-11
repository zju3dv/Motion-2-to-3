import torch
from pytorch3d.transforms import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
)
from hmr4d.utils.matrix import slerp
from hmr4d.dataset.supermotion.collate import pad_to_max_len
from hmr4d.utils.geo.optim_rotation import R_from_wy


def get_window_startends(seq_length, max_L, overlap):
    startends = []
    for s in range(0, seq_length, max_L - overlap):
        if s + max_L >= seq_length:
            startends.append((s, seq_length))
            break
        startends.append((s, s + max_L))
    return startends


def split_pad_batch(target, max_L, startends=None, no_ovelap=False):
    if startends is None:  # Non-sliding window
        assert no_ovelap
        startends = get_window_startends(target.shape[0], max_L, overlap=0)
    target = [pad_to_max_len(target[s:e], max_L) for s, e in startends]
    return torch.stack(target, dim=0)  # (B', L, ...)


def blend_overlap(last_overlap, next_overlap, vtype="vec"):
    assert last_overlap.shape == next_overlap.shape
    overlap_L = last_overlap.shape[0]
    device = last_overlap.device

    if vtype == "vec":
        weight_last = torch.linspace(0, 1, overlap_L, device=device)
        weight_next = torch.linspace(1, 0, overlap_L, device=device)
        last_overlap = torch.einsum("l, l ... -> l ...", weight_last, last_overlap)
        next_overlap = torch.einsum("l, l ... -> l ...", weight_next, next_overlap)
        blended_overlap = last_overlap + next_overlap
    elif vtype == "aa_j3":

        J = last_overlap.shape[-1] // 3
        shape_0 = last_overlap.shape[:-1]
        last_aa = last_overlap.reshape(*shape_0, J, 3)
        last_quat = axis_angle_to_quaternion(last_aa)[..., [1, 2, 3, 0]]
        next_aa = next_overlap.reshape(*shape_0, J, 3)
        next_quat = axis_angle_to_quaternion(next_aa)[..., [1, 2, 3, 0]]

        t = torch.linspace(0, 1, overlap_L, device=device)
        for _ in range(len(last_quat.shape) - 1):
            t = t.unsqueeze(-1)
        blended_quat = slerp(last_quat, next_quat, t)[..., [3, 0, 1, 2]]
        blended_overlap = quaternion_to_axis_angle(blended_quat).reshape(*shape_0, -1)
    else:
        raise NotImplementedError

    return blended_overlap


def slide_merge(x, startends, vtype="vec", do_blend=True):
    """
    x: (B, L, *), slide merge L dimension
    startends: List of tuple
    """
    assert len(x) == len(startends)
    assert startends[0][0] == 0
    last_end = 0
    seq_length = startends[-1][1]
    x_merged = torch.zeros(seq_length, *x[0].shape[1:], device=x[0].device)  # e.g. (max_L, C)
    for w, (s, e) in enumerate(startends):
        if last_end == s:
            x_merged[s:e] = x[w][: e - s]
        else:
            assert last_end > s
            assert last_end - s < e - s, "no-overlap not exists"
            # assign the no-overlap part directly
            x_merged[last_end:e] = x[w][last_end - s : e - s]
            if do_blend:  # linear blends the overlap part
                last_overlap = x_merged[s:last_end].clone()
                next_overlap = x[w][: last_end - s].clone()
                x_merged[s:last_end] = blend_overlap(last_overlap, next_overlap, vtype)
            else:
                x_merged[s:last_end] = x[w][: last_end - s]
        last_end = e
    return x_merged


def slide_merge_root_aa_ayfz(x, startends):
    assert len(x) == len(startends)
    assert startends[0][0] == 0
    last_end = 0
    seq_length = startends[-1][1]
    x_merged = torch.zeros(seq_length, *x[0].shape[1:], device=x[0].device)  # e.g. (max_L, C)

    if False:  # Visualize y-axis angles
        x_euler = matrix_to_euler_angles(axis_angle_to_matrix(x), "YXZ")
        x_y_angle = x_euler[..., 0]
        # Draw the line of x_y_angle with plt, and save to disk
        import time
        from pathlib import Path

        time_postfix = str(int(time.time()))
        out_fn = f"tmp_angle/{time_postfix}.png"
        Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        plt.cla()
        data_to_plot = x_y_angle.cpu().numpy()
        for d in data_to_plot:
            plt.plot(d)
        plt.savefig(out_fn)
        return

    for w, (s, e) in enumerate(startends):
        if last_end == s:
            x_merged[s:e] = x[w][: e - s]
        else:
            assert last_end > s
            assert last_end - s < e - s, "no-overlap not exists"

            # The current version uses concatenation instead of blending
            # last_window_end = s + (last_end - s) // 2  # (transformed next_window will be putted from this frame)
            last_window_end = s
            next_window_start_local = last_window_end - s

            last_R = axis_angle_to_matrix(x_merged[last_window_end].clone())  # (3, 3)
            next_R = axis_angle_to_matrix(x[w][next_window_start_local].clone())  # (3, 3)

            with torch.enable_grad():
                w_yaxis = torch.tensor([0.0]).cuda().requires_grad_()
                optimizer = torch.optim.SGD([w_yaxis], lr=0.01)
                last_loss = torch.inf
                for i in range(100):
                    R_next2last = R_from_wy(w_yaxis, use_sincos=True)
                    loss = torch.norm(R_next2last @ next_R - last_R, "fro")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # exit if loss converge
                    if (last_loss - loss).abs() < 1e-5:
                        break
                    last_loss = loss
                    # print(f"{i} loss: {loss.item()}")
                # print(i)

            R_next2last = R_from_wy(w_yaxis.detach(), use_sincos=True)
            next_window_R = axis_angle_to_matrix(x[w][next_window_start_local : e - s])
            next_window_R = R_next2last @ next_window_R
            next_window_aa = matrix_to_axis_angle(next_window_R)

            # assign the next window part directly
            x_merged[last_window_end:e] = next_window_aa

        last_end = e
    return x_merged
