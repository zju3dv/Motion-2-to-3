import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from einops import repeat, rearrange
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.phc.pos2smpl import convert_pos_to_smpl
from scipy.ndimage._filters import _gaussian_kernel1d
import trimesh


MODEL = "single_fid308"
MODEL = "single_cb"
# MODEL = "single_scratch"
# MODEL = "single_3view"
# MODEL = "single_5view"
MODEL = "single_oneV100_3view"
# MODEL = "single_oneV100_5view"
# MODEL = "mdm"
MODEL = "seed42_ele20"
subset= "t2m"
path = f"./outputs/dumped_t2m_{MODEL}"
print(f"Load from {path}")
saved_motions = os.listdir(path)
saved_motions.sort()
saved_motions = [os.path.join(path, p) for p in saved_motions if "pth" in p]
selected_inds = [_ for _ in range(len(saved_motions))]

for s_i in tqdm(selected_inds[:100]):
    # if s_i != 3212:
    #     continue

    p = saved_motions[s_i]
    motion = torch.load(p, map_location="cpu")
    text = motion["text"]
    length = motion["length"]
    joints = motion["pred"]
    joints = joints[:length]
    # if "walk" not in text:
    #     continue
    # if len(text.split(" ")) < 15:
    #     continue
    # joints = gaussian_filter(joints)
    try:
        wis3d = make_wis3d(output_dir=f"outputs/wis3d_debug_{subset}", name=f"{s_i:04}")
        add_motion_as_lines(joints, wis3d, name=f"{MODEL}_{text}")
    except Exception as e:
        print(e)
    