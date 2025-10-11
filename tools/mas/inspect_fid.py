import torch
import numpy as np
from pathlib import Path
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.net_utils import repeat_to_max_len
from hmr4d.utils.hml3d.metric import calculate_activation_statistics_np, calculate_frechet_distance_np

# Features
gt = torch.load("./hmr4d/utils/hml3d/gt_stat.pth")
all_res = torch.load("./outputs/tmp_dump_fid.pt", map_location="cpu")


def update_saved_files_for_visualization(saved_files):
    """Make everything to 300 frames, with repeating last frames"""
    for k, data in saved_files.items():
        pred = data["pred"]
        gt = data["gt"]
        length = data["length"]
        # if not (pred.shape[0] == gt.shape[0] and pred.shape[0] == length):
        #     print(f"[{k}] pred {pred.shape}, gt {gt.shape}, length {length}")
        pred = repeat_to_max_len(pred[:length], 300, 0)
        gt = repeat_to_max_len(gt[:length], 300, 0)
        data["pred"] = pred
        data["gt"] = gt


update_saved_files_for_visualization(all_res)

#############################################
from hmr4d.utils.eval.torch_fid import compute_stats, compute_fid

# numpy implementation
# mu, cov = calculate_activation_statistics_np(all_emb.numpy())
# fid = calculate_frechet_distance_np(gt["mu"], gt["cov"], mu, cov)

all_emb = torch.cat([v["emb"] for v in all_res.values()], dim=0)
pred_emb = all_emb.clone().requires_grad_()  # (N, 512)
gt_mu_ts = torch.from_numpy(gt["mu"]).float()  # (512,)
gt_cov_ts = torch.from_numpy(gt["cov"]).float()  # (512, 512)

mu, cov = compute_stats(pred_emb)
fid = compute_fid(gt_mu_ts, gt_cov_ts, mu, cov)
fid.backward()

# print
# 计算每个维度上的梯度绝对值的平均值
dim_grad_abs_mean = pred_emb.grad.abs().mean(dim=0)

# 按维度梯度平均值排序，找出影响最大的维度
key_dim_order = torch.argsort(dim_grad_abs_mean, descending=True).tolist()

# 最关键的维度
most_important_dims = key_dim_order[:10]
print("Most important dimensions:", most_important_dims)

# grad_abs_sum = pred_emb.grad.abs().sum(-1)
grad_abs_sum = pred_emb.grad[:, most_important_dims].sum(-1)
grad_desc_order = torch.argsort(grad_abs_sum, descending=True).tolist()
worst_inds = grad_desc_order[-25:]
best_inds = grad_desc_order[:5]
print("worst:", worst_inds)
print("best:", best_inds)

############################################

for ind in worst_inds:
    key = f"{ind:05d}"

    pred = all_res[key]["pred"]
    gt = all_res[key]["gt"]
    length = all_res[key]["length"]
    text = all_res[key]["text"]
    text = text.replace("/", "_")
    text = text[:200]
    wis3d = make_wis3d(name="worst-" + key)
    add_motion_as_lines(gt, wis3d, name=f"gt ::: {text}", const_color="green")
    add_motion_as_lines(pred, wis3d, name="pred", const_color="blue")

for ind in best_inds:
    key = f"{ind:05d}"

    pred = all_res[key]["pred"]
    gt = all_res[key]["gt"]
    length = all_res[key]["length"]
    text = all_res[key]["text"]
    text = text.replace("/", "_")
    text = text[:200]
    wis3d = make_wis3d(name="best-" + key)
    add_motion_as_lines(gt, wis3d, name=f"gt ::: {text}", const_color="green")
    add_motion_as_lines(pred, wis3d, name="pred", const_color="blue")
