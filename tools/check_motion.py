import torch
import os

from hmr4d.utils.o3d_utils import o3d_skeleton_animation

ROOT = "./inputs/amass/smpl22_joints3d_neutral.pth"
motions = torch.load(ROOT)

ROOT2 = "./inputs/amass/smpl22_joints3d_neutral_v2.pth"
motions2 = torch.load(ROOT2)
for k in motions2.keys():
    m1 = motions[k]
    m2 = motions2[k]
    import ipdb

    ipdb.set_trace()

prefix = "inputs/amass/smplhg_raw/"

# test idx=0, 4822: ./pose_data/BioMotionLab_NTroje/rub006/0002_treadmill_slow_poses.npy
k = os.path.join(prefix, "BioMotionLab_NTroje/rub006/0002_treadmill_slow_poses.npz")
ks = list(motions.keys())
ks = [k for k in ks if "Minputs" not in k]  # filter mirror data
ks_ = [k for k in ks if "BioMotionLab_NTroje" in k]
for k in ks_:
    print(k)
    joints3d = motions[k]

    o3d_skeleton_animation(joints3d)
