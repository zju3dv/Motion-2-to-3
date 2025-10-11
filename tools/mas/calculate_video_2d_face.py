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
    compute_camera_facing,
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


yaml_file = "hmr4d/configs/data/motion2d/IDEA400_2d_nr.yaml"
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
#K_adj = []
incam_root3d = []
face_t_list = [[0,0,0,0],[0,0,0,0]] # front, right, back, left
# face_dg_per_frame is a list of 36 zeros
face_dg_per_frame = [0] * 72
frames_num = 0
# 36 categories
for batch in tqdm(dataloader, total=len(dataloader)):
    assert batch["gt_motion"].shape[0] == 1
    l = batch["length"][0]
    all_motion2d.append(batch["gt_motion2d"][0, :l].flatten(1))
    #print(f"i: {i}, l: {l}")
    #print(f"shape: {batch['gt_motion'].shape}")
    face, face_t = compute_camera_facing(batch["gt_motion"][0, :l])
    for f in face:
        face_dg_per_frame[int(f/5) + 36] += 1 # -180 is category 0

    face_t_list[0][int(face_t[0])] += 1
    face_t_list[1][int(face_t[1])] += 1
    #K_adj.append(batch["K_adj"][0])
    incam_root3d.append(batch["gt_motion"][0, :l, 0].norm(dim=-1))
    i += 1
    frames_num += int(l)
    #if(i == 5):
    #    break
incam_root3d = torch.cat(incam_root3d, dim=0)

face_portion = [f/sum(face_t_list[0]) for f in face_t_list[0]]
face_portion2 = [f/sum(face_t_list[1]) for f in face_t_list[1]]
#print("front, right, back, left")
#print(f"face portion: {face_portion}, face_portion2:{face_portion2}")
print("front is 45 degree to 135 degree")
print("face_dg_per_frame:")
face_dg_portion = [f/frames_num for f in face_dg_per_frame]
face_dg_dict = {}
for i in range(72):
    face_dg_dict.update({i*5-180: face_dg_portion[i]})
print(face_dg_dict)
print(face_dg_portion)
# Maybe we can use face labels
labels = ['front', 'right', 'back', 'left']
print("can do more analysis on face_t_list, face_dg_per_frame")
import ipdb

ipdb.set_trace()
