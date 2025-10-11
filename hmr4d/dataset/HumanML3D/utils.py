import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import copy
import torch.nn.functional as F


def load_motion_files(path, humanact_path, index_path):
    index_file = pd.read_csv(index_path)
    source_paths = []
    new_names = []
    start_frames = []
    end_frames = []
    for i in range(index_file.shape[0]):
        source_path = index_file.loc[i]["source_path"]
        new_name = index_file.loc[i]["new_name"]
        start_frame = index_file.loc[i]["start_frame"]
        end_frame = index_file.loc[i]["end_frame"]
        if "humanact12" in source_path:
            source_path = source_path.replace("./pose_data/humanact12", humanact_path)
        else:
            source_path = source_path.replace("./pose_data", path)
            source_path = source_path.replace(".npy", ".npz")
        source_paths.append(source_path)
        new_names.append(new_name)
        start_frames.append(start_frame)
        end_frames.append(end_frame)

    return source_paths, new_names, start_frames, end_frames


def swap_left_right(data):
    # ignore hands
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.clone()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    return data


def upsample_motion(motion, original_fps, target_fps):
    motion_shape_len = len(motion.shape)
    if motion_shape_len == 2:
        motion = motion[:, None]

    N, J, C = motion.shape  # N: 帧数，J: 关节点数，C: 坐标数（通常为3）

    # 计算目标帧数
    scale_factor = 1.0 * target_fps / original_fps
    target_N = int(N * scale_factor)

    # 调整数据维度以适应interpolate函数
    # 将数据重构为(J, C, N)，其中C是通道数，即原来的J
    motion = motion.permute(1, 2, 0)

    # 使用线性插值进行上采样
    upsampled_motion = F.interpolate(motion, size=target_N, mode="linear", align_corners=True)

    # 恢复原始的维度顺序
    upsampled_motion = upsampled_motion.permute(2, 0, 1)

    if motion_shape_len == 2:
        upsampled_motion = upsampled_motion[:, 0]
    return upsampled_motion


def resample_motion_fps(motion, target_length):
    N, J, C = motion.shape  # N: 帧数，J: 关节点数，C: 坐标数（通常为3）

    motion = motion.permute(1, 2, 0)

    upsampled_motion = F.interpolate(motion, size=target_length, mode="linear", align_corners=True)

    upsampled_motion = upsampled_motion.permute(2, 0, 1)
    return upsampled_motion
