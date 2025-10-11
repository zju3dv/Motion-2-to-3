from hmr4d.utils.plt_utils import plt_skeleton_animation
from hmr4d.dataset.motionx.utils import normalize_kp_2d, adjust_K, estimate_focal_length, generate_camera_intrinsics
import pickle
import json
import os
import joblib
import torch

def load_pkl(pkl_path):
    data = joblib.load(pkl_path)
    if len(data.keys()) == 0:
        print(f"{pkl_path} does not have valid data!")
        return None
    x = data[0]
    if len(x.keys()) == 0:
        print(f"{pkl_path} does not have valid data!")
        return None
    return x

#for vid_path in vid_list.keys():
for i in range(1):
    #output_path = vid_list[vid_path]
    
    #pkl_path = f"./result/{output_path}/ezi_output.pkl"
    #pkl_path = f"../../../../dy_data/MESH-GRAPHORMER/grx/MyWHAM/results/idea400_light/Act_cute_and_sitting_at_the_same_time_clip1/wham_output.pkl"
    #pkl_path = f"./ezi_output_v1.pkl"
    pkl_path = f"./tools/exp_ezi/fair_cook_05_6-01_ezi_output.pkl"
    if os.path.exists(pkl_path):
        data = load_pkl(pkl_path)
        for k in data.keys():
            print(k)
            #print(data[k].dtype)
            if 'vid' in k :
                #print(data[k])
                continue
            if 'motion2d_points' in k:
                print(data[k].shape)
                i_motion2d_1 = torch.tensor(data[k][:,:,:2], dtype=torch.float32) # (F, J, 2)
                #i_motion2d_1 = torch.tensor(data[k], dtype=torch.float32)  # (F, J, 2)
                normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d_1)  # (F, J, 2)
                print(f"i_motion2d: {i_motion2d_1}")
                plt_skeleton_animation(normed_motion2d, skeleton_type="smpl")