import os
import numpy as np
import torch
import json
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
import joblib


def load_motion_files(directory):
    sequences = []
    for root, dirs, files in os.walk(directory, followlinks=True):
        # Check if 'wham_output.pkl' file exists in the current path
        # root: inputs/wham_video/idea400_video/new_ntu_thu_15fps/subset_0005/A251-A265____MVI_9720
        # dirs: []
        # files: ["wham_output.pkl"]

        if "ezi_output_merged.pkl" in files:
            sequences.append(root)

    return sequences


def load_pkl(pkl_path):
    data = joblib.load(pkl_path)
    if len(data.keys()) == 0:
        print(f"{pkl_path} does not have valid data!")
        return None
    # if len(data.keys()) > 1:
    #     print(f"{root} have {len(data.keys())} characters!")
    # Assume only one character
    #print(data.keys())
    x = data
    if len(x.keys()) == 0:
        print(f"{pkl_path} does not have valid data!")
        return None
    return x


def smpl_fk(smpl_model, body_pose, betas, global_orient=None, transl=None):
    """
    Args:
        body_pose: (B, L, 63)
        betas: (B, L, 10)
        global_orient: (B, L, 3)
    Returns:
        joints: (B, L, 22, 3)
    """
    flag = False
    if len(body_pose.shape) == 2:
        body_pose = body_pose[None]
        betas = betas[None]
        if global_orient is not None:
            global_orient = global_orient[None]
        if transl is not None:
            transl = transl[None]
        flag = True

    B, L = body_pose.shape[:2]
    if global_orient is None:
        global_orient = torch.zeros((B, L, 3), device=body_pose.device)
    aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
    rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)
    parents = smpl_model.bm.parents[:22]

    skeleton = smpl_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
    local_skeleton = skeleton - skeleton[:, :, parents]
    local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

    if transl is not None:
        local_skeleton[..., 0, :] += transl  # B, L, 22, 3

    mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
    fk_mat = matrix.forward_kinematics(mat, parents)  # B, L, 22, 4, 4
    joints = matrix.get_position(fk_mat)  # B, L, 22, 3
    if flag:
        joints = joints[0]

    return joints


def augment_motion_files(motion_files, is_notegoexo=False, N=5):
    augmented_files = {}
    ori_keys = list(motion_files.keys())
    merge_n = []
    
    # 第一个循环：截断每个段的 `incam_joints` 和 `bboxs`
    for key in ori_keys:
        segment = motion_files[key]
        # 根据 `pred_scores.shape[0]` 截断 `incam_joints` 和 `bboxs`
        segment["incam_joints"] = segment["incam_joints"][:segment["pred_scores"].shape[0]]
        segment["bboxs"] = segment["bboxs"][:segment["pred_scores"].shape[0]]
        segment["length"] = segment["pred_scores"].shape[0]
        
        # 保存截断后的段
        augmented_files[key] = segment
    
    if is_notegoexo:
        return augmented_files  

    # 第二个循环：寻找相邻段并进行多次合并
    for key in ori_keys:
        combined_incam_joints = augmented_files[key]["incam_joints"]
        combined_bboxs = augmented_files[key]["bboxs"]
        combined_pred_scores = augmented_files[key]["pred_scores"]
        combined_name = [augmented_files[key]["name"]]
        combined_video_path = [augmented_files[key]["video_path"]]

        # 当前合并段的key，用于生成新的合并名称
        current_key = key
        frame_num = int(current_key.rsplit('_', 1)[1].split('/')[0])

        for i in range(1, N):  # 从1开始，因为第0段已经是基段
            # 计算下一个相邻段的 key
            next_frame_num = f"{frame_num + i:04d}"
            base_name, cam_id = current_key.rsplit('_', 1)[0], current_key.split('/')[-1]
            adjacent_key = f"{base_name}_{next_frame_num}/{cam_id}"

            # 若找不到相邻段或达到N，则停止合并
            if adjacent_key not in augmented_files:
                break

            # 追加相邻段数据
            seg = augmented_files[adjacent_key]
            combined_incam_joints = torch.cat([combined_incam_joints, seg["incam_joints"]], dim=0)
            combined_bboxs = torch.cat([combined_bboxs, seg["bboxs"]], dim=0)
            combined_pred_scores = torch.cat([combined_pred_scores, seg["pred_scores"]], dim=0)
            combined_name.append(seg["name"])
            combined_video_path.append(seg["video_path"])

            if combined_incam_joints.shape[0] > 300:
                break

        # 合并结果存储到 `augmented_files`，并加上后缀
        if len(combined_name) > 1:
            augmented_files[f"{key}!aug"] = {
                "length": combined_incam_joints.shape[0],
                "bboxs": combined_bboxs,
                "incam_joints": combined_incam_joints,
                "pred_scores": combined_pred_scores,
                "name": combined_name,
                "video_path": combined_video_path,
            }
            merge_n.append(len(combined_name))
        else:
            merge_n.append(1)

    avg_merge_n = sum(merge_n) / len(merge_n)
    Log.info(f"Augmentation: requrested merge {N}, final average merge number: {avg_merge_n}")
    
    return augmented_files