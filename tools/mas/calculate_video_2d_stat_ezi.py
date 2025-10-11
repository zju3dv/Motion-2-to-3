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

def large_batch_mean_var(x, batch_size):
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

    return mean_sum, sq_sum, total_samples

yaml_file = "hmr4d/configs/data/motion2d/All_200h.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/Motion_xpp_music.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/EgoExo_Dance_2d_nr.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/EgoExo_EZI_music.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/Music_goodlower.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/Music_Mix.yaml"
#yaml_file = "hmr4d/configs/data/motion2d/All_cur.yaml"
data_opts = OmegaConf.load(yaml_file).dataset_opts

data_train_opts = data_opts.train
for k in data_opts.keys():
    if k not in ["train", "val"]:
        for opt in data_train_opts:
            if k in opt.keys():
                opt[k] = data_opts[k]

print(data_train_opts)
dataset = ConcatDataset([instantiate(d) for d in data_train_opts])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

print("Calculating statistics")


all_motion2d = []
all_mask = []
i = 0
mean_sum = None #torch.zeros(x[0].size(1)).cuda()
sq_sum = None #torch.zeros(x[0].size(1)).cuda()
total_samples = 0

for batch in tqdm(dataloader, total=len(dataloader)):
    #assert batch["gt_motion"].shape[0] == 1 
    l = batch["length"][0]
    if l == False:
        continue
    all_motion2d.append(batch["gt_motion2d"][0, :l].flatten(1))
    if mean_sum == None:
        mean_sum = torch.zeros(all_motion2d[-1].size(1)).cuda()
        sq_sum = torch.zeros(all_motion2d[-1].size(1)).cuda()

    if len(all_motion2d) > 10000:
        cur_mean_sum, cur_sq_sum, cur_total_samples = large_batch_mean_var(all_motion2d, 10000)
        mean_sum += cur_mean_sum
        sq_sum += cur_sq_sum
        total_samples += cur_total_samples
        all_motion2d = [] # clear
    #all_mask.append(batch["mask"][0, :l].flatten(1))
    i += 1

if len(all_motion2d) > 0:
    cur_mean_sum, cur_sq_sum, cur_total_samples = large_batch_mean_var(all_motion2d, 10000)
    mean_sum += cur_mean_sum
    sq_sum += cur_sq_sum
    total_samples += cur_total_samples

'''Large batches'''
mean = mean_sum / total_samples
var = (sq_sum / total_samples) - (mean**2)

#mean, var = batch_mean_var(all_motion2d, 10000) # * Small batches
#mean_mask, var_mask = batch_mean_var(all_mask, 10000)
#print(mean_mask)

# find var might be < 0 but very close to 0 due to numerical
var = var.abs()
std = var**0.5
std[std < 1e-4] = 1.0
statistics = {"mean": mean.cpu(), "std": std.cpu()}
print("Please manually save the statistics to the statistics.py file")
import ipdb

ipdb.set_trace()
