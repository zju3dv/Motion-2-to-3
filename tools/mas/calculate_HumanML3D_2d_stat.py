import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from hydra.utils import instantiate


from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset

from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
torch.multiprocessing.set_sharing_strategy("file_system")


def batch_mean_var(x, batch_size):
    # x: List (Tensor (1, C))
    n_samples = len(x)
    n_batches = (n_samples + batch_size - 1) // batch_size  # 确保覆盖所有数据

    mean_sum = torch.zeros(x[0].size(1)).cuda()
    sq_sum = torch.zeros(x[0].size(1)).cuda()
    total_samples = 0

    for i in tqdm(range(n_batches)):
        batch = x[i * batch_size : min((i + 1) * batch_size, n_samples)]
        batch = torch.cat(batch, dim=0).cuda()
        batch_mean = batch.mean(0)
        batch_sq_mean = batch.pow(2).mean(0)

        mean_sum += batch_mean * batch.size(0)
        sq_sum += batch_sq_mean * batch.size(0)
        total_samples += batch.size(0)

    mean = mean_sum / total_samples
    var = (sq_sum / total_samples) - (mean**2)
    return mean, var


yaml_file = "hmr4d/configs/data/motion2d/HumanML3D_2d_nr.yaml"
data_opts = OmegaConf.load(yaml_file).dataset_opts

data_train_opts = data_opts.train
for k in data_opts.keys():
    if k not in ["train", "val"]:
        for opt in data_train_opts:
            if k in opt.keys():
                opt[k] = data_opts[k]

print(data_train_opts)
dataset = ConcatDataset([instantiate(d) for d in data_train_opts])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

all_motion2d = []
i = 0
for batch in tqdm(dataloader, total=len(dataloader)):
    assert batch["gt_motion"].shape[0] == 1
    l = batch["length"][0]
    all_motion2d.append(batch["gt_motion2d"][0, :l].flatten(1))
    trans_mat = compute_T_ayf2az(batch["gt_motion"][0, :l])
    i += 1
    if(i == 150):
        break
mean, var = batch_mean_var(all_motion2d, 10000)

# find var might be < 0 but very close to 0 due to numerical
var = var.abs()
std = var**0.5
std[std < 1e-4] = 1.0
statistics = {"mean": mean.cpu(), "std": std.cpu()}
print("Please manually save the statistics to the statistics.py file")
import ipdb

ipdb.set_trace()
