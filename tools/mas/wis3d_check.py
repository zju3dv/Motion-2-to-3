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


# MODEL = "ours"
MODEL = "mdm"
# MODEL = "singlemodel"
# MODEL = "motionclip"
# MODEL = "motionbert"
# MODEL = "avatarclip"
# MODEL = "mas"
# subset= "comp_basketball"
subset= "mixed"
path = f"./outputs/dumped_{subset}_{MODEL}"
print(f"Load from {path}")
saved_motions = os.listdir(path)
saved_motions.sort()
saved_motions = [os.path.join(path, p) for p in saved_motions if "pth" in p]
selected_inds = [_ for _ in range(len(saved_motions))]

for s_i in tqdm(selected_inds[:100]):
    p = saved_motions[s_i]
    motion = torch.load(p, map_location="cpu")
    text = motion["text"]
    length = motion["length"]
    joints = motion["pred"]
    joints = joints[:length]
    # joints = gaussian_filter(joints)
    wis3d = make_wis3d(output_dir=f"outputs/wis3d_debug_{subset}", name=f"{s_i:03}")
    add_motion_as_lines(joints, wis3d, name=f"{MODEL}_{text}")
    