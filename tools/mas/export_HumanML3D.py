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


yaml_file = "hmr4d/configs/data/motion2dexport/HumanML3D_2d.yaml"
data_train_opts = OmegaConf.load(yaml_file).dataset_opts.train

# This process will use all data and convert them to joints3d
print(data_train_opts)
dataset = ConcatDataset([instantiate(d) for d in data_train_opts])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

all_joints3d = {}
i = 0
for batch in tqdm(dataloader, total=len(dataloader)):
    assert batch["gt_motion"].shape[0] == 1
    name = batch["name"][0]
    new_name = batch["new_name"][0]
    if batch["is_mirror"][0]:
        new_name = "M" + new_name
    all_joints3d[new_name] = {
        "joints3d": batch["gt_motion"][0].numpy(),
        "name": name,
    }
    i += 1

os.makedirs("inputs/hml3d", exist_ok=True)
torch.save(all_joints3d, "inputs/hml3d/joints3d.pth")
print(f"Save successfully at inputs/hml3d/joints3d.pth")
