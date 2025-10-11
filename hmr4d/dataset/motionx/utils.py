# process kp2d is from WHAM (https://github.com/yohanshin/WHAM/blob/train/lib/data/datasets/amass.py#L65)
# Modify it to center at root
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import copy
import cv2
import torch.nn.functional as F


def load_motion_files(path):
    pathlist = Path(path).rglob("*.npz")
    # convert generator to list
    pathlist = [str(p) for p in pathlist]
    return pathlist


def load_motion_and_text(base_motion_path, base_text_path):
    data = {}
    
    # 遍历motion数据的子集文件夹
    for subset in os.listdir(base_motion_path):
        motion_subset_path = os.path.join(base_motion_path, subset)
        text_subset_path = os.path.join(base_text_path, subset)

        if os.path.isdir(motion_subset_path) and os.path.isdir(text_subset_path):
            # 遍历每个subset中的.npy文件
            for file in os.listdir(motion_subset_path):
                if file.endswith('.npy'):
                    motion_file_path = os.path.join(motion_subset_path, file)
                    text_file_path = os.path.join(text_subset_path, file.replace('.npy', '.txt'))

                    # 读取motion和text
                    if os.path.exists(text_file_path):
                        motion = np.load(motion_file_path)
                        with open(text_file_path, 'r') as f:
                            text = f.read().strip()
                        
                        # 存储在字典中
                        data[subset+ "/" + file[:-4]] = {"motion": motion, "text": text}

    return data


def load_mixed(base_motion_path, base_text_path):
    mixed_text_path = f"./inputs/motionx/motionx_seq_text_v1.1/mixed_test_seq_names.json"
    with open(mixed_text_path, "r") as file:
        test_seq_names = json.load(file)
    data = {}
    for k in test_seq_names:
        motion_path = os.path.join(base_motion_path, k + ".npy")
        text_path = os.path.join(base_text_path, k + ".txt")
        motion = np.load(motion_path)
        with open(text_path, 'r') as f:
            text = f.read().strip()
        data[k] = {"motion": motion, "text": text}

    return data


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.0]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y  # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_keypoints(kp_2d, bbox, patch_width, patch_height):
    center_x, center_y, scale = bbox[:3]
    width = height = scale * 200
    # scale, rot = 1.2, 0
    # scale, rot = 1.0, 0

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale=1.0,
        rot=0.0,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans


def normalize_keypoints_to_patch(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0) / (2 * ratio)

    return kp_2d


def normalize_keypoints_to_image(x, res):
    res = res.to(x.device)
    scale = res.max(-1)[0].reshape(-1)
    mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
    x = 2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - mean.reshape(
        *[1 for i in range(len(x.shape[1:-1]))], -1
    )
    return x


def compute_bbox_from_keypoints(X, do_augment=False, mask=None, not_moving=False):
    def smooth_bbox(bb):
        # Smooth bounding box detection
        import scipy.signal as signal

        smoothed = np.array([signal.medfilt(param, int(30 / 2)) for param in bb])
        return smoothed

    def do_augmentation(scale_factor=0.2, trans_factor=0.05):
        _scaleFactor = np.random.uniform(1.0 - scale_factor, 1.2 + scale_factor)
        _trans_x = np.random.uniform(-trans_factor, trans_factor)
        _trans_y = np.random.uniform(-trans_factor, trans_factor)

        return _scaleFactor, _trans_x, _trans_y

    if do_augment:
        scaleFactor, trans_x, trans_y = do_augmentation()
    else:
        scaleFactor, trans_x, trans_y = 1.2, 0.0, 0.0

    if mask is None:
        bbox = [X[:, :, 0].min(-1)[0], X[:, :, 1].min(-1)[0], X[:, :, 0].max(-1)[0], X[:, :, 1].max(-1)[0]]
    else:
        bbox = []
        for x, _mask in zip(X, mask):
            if _mask.sum() > 10:
                _mask[:] = False
            _bbox = [x[~_mask, 0].min(-1)[0], x[~_mask, 1].min(-1)[0], x[~_mask, 0].max(-1)[0], x[~_mask, 1].max(-1)[0]]
            bbox.append(_bbox)
        bbox = torch.tensor(bbox).T

    #### original ####
    # cx, cy = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    # bbox_w = bbox[2] - bbox[0]
    # bbox_h = bbox[3] - bbox[1]
    #### original ####

    #### modify to root center ####
    cx, cy = X[:, 0, 0], X[:, 0, 1]
    bbox_w = torch.maximum(cx - bbox[0], bbox[2] - cx) * 2
    bbox_h = torch.maximum(cy - bbox[1], bbox[3] - cy) * 2
    if not_moving:
        # a box for the whole sequence
        bbox_w = bbox_w.max()[None].expand_as(bbox_w)
        bbox_h = bbox_h.max()[None].expand_as(bbox_h)

    bbox_size = torch.stack((bbox_w, bbox_h)).max(0)[0]
    scale = bbox_size * scaleFactor
    bbox = torch.stack((cx + trans_x * scale, cy + trans_y * scale, scale / 200))

    if do_augment:
        bbox = torch.from_numpy(smooth_bbox(bbox.numpy()))

    return bbox.T


def bbox_normalization(kp_2d, bbox, patch_width, patch_height):
    to_torch = False
    if isinstance(kp_2d, torch.Tensor):
        to_torch = True
        kp_2d = kp_2d.numpy()
        bbox = bbox.numpy()

    out_kp_2d = np.zeros_like(kp_2d)
    norm_kp_2d = np.zeros_like(kp_2d)
    for idx in range(len(out_kp_2d)):
        out_kp_2d[idx] = transform_keypoints(copy.deepcopy(kp_2d[idx]), bbox[idx][:3], patch_width, patch_height)[0]
        norm_kp_2d[idx] = normalize_keypoints_to_patch(out_kp_2d[idx], patch_width)

    if to_torch:
        out_kp_2d = torch.from_numpy(out_kp_2d)
        norm_kp_2d = torch.from_numpy(norm_kp_2d)

    return norm_kp_2d, out_kp_2d


def normalize_kp_2d(kp_2d, patch_width=224, patch_height=224, bbox=None, mask=None, not_moving=False, multiview=False, randselect=True, scale=None):
    """_summary_
        kpts from [-1, 1] (not extactly)

    Args:
        kp_2d (tensor): (F, J, 2)
        patch_width (int, optional): _description_. Defaults to 224.
        patch_height (int, optional): _description_. Defaults to 224.
        bbox (_type_, optional): _description_. Defaults to None.
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if multiview:
        if bbox is None:
            bbox = []
            max_bbox = None
            for i in range(kp_2d.shape[0]):
                bbox_i = compute_bbox_from_keypoints(kp_2d[i], do_augment=False, mask=mask, not_moving=not_moving)
                bbox.append(bbox_i)
                ### max box
                if max_bbox is None:
                    max_bbox = bbox_i 
                else:
                    max_bbox = torch.maximum(max_bbox, bbox_i)
                #########
            if randselect:
                ind = torch.randint(0, kp_2d.shape[0], (1,))
                bbox = bbox[ind]
            else:
                bbox = max_bbox
            
            if scale is not None:
                bbox[:, -1] = scale
        
        norm_kp_2d = []
        out_kp_2d = []
        for i in range(kp_2d.shape[0]):
            norm_kp_2d_i, out_kp_2d_i = bbox_normalization(kp_2d[i], bbox, patch_width, patch_height)
            norm_kp_2d.append(norm_kp_2d_i)
            out_kp_2d.append(out_kp_2d_i)
        norm_kp_2d = torch.stack(norm_kp_2d, dim=0)
        out_kp_2d = torch.stack(out_kp_2d, dim=0)
    else:
        if bbox is None:
            bbox = compute_bbox_from_keypoints(kp_2d, do_augment=False, mask=mask, not_moving=not_moving)
            if scale is not None:
                bbox[:, -1] = scale

        #if mask is not None:
        #    kp_2d = kp_2d * mask.unsqueeze(-1) #Maked out keypoints to 0

        norm_kp_2d, out_kp_2d = bbox_normalization(kp_2d, bbox, patch_width, patch_height)

    return norm_kp_2d, out_kp_2d, bbox


def adjust_K(K, bbox, patch_width=224, patch_height=224):
    """
    Adjust the camera matrix K based on the bounding box (bbox) information.
    This version also corrects for translation to ensure the patch is centered on the bbox center.

    Args:
        K: Original camera matrix (B, 3x3).
        bbox: Bounding box information, (B, 3) expected to be [cx, cy, scale].
        patch_width: Width of the image or region to project onto.
        patch_height: Height of the image or region to project onto.

    Returns:
        Adjusted camera matrix.
    """
    cx, cy, scale = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    K_adj = K.clone()
    scale = scale * 200  # Adjust the scale as per the bbox scale
    patch_scale = max(patch_width, patch_height)

    # Adjust focal length based on the scale
    scale_factor = patch_scale / scale
    K_adj[:, 0, 0] *= scale_factor  # Adjusting focal length in x direction
    K_adj[:, 1, 1] *= scale_factor  # Adjusting focal length in y direction

    # Correct translation to center the bbox in the patch
    # This is done by adjusting the principal point in the camera matrix
    K_adj[:, 0, 2] = patch_width / 2.0 - (cx - K[:, 0, 2]) * scale_factor
    K_adj[:, 1, 2] = patch_height / 2.0 - (cy - K[:, 1, 2]) * scale_factor

    return K_adj


# From CLIFF
def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree


def generate_camera_intrinsics(img_h, img_w):
    """
    Generate camera intrinsics matrix K given image height and width.

    Args:
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        K: Camera intrinsics matrix (3x3 tensor).
    """
    focal_length = estimate_focal_length(img_h, img_w)

    # Assuming the optical center is at the center of the image
    c_x = img_w / 2
    c_y = img_h / 2

    # Constructing the camera intrinsics matrix
    K = torch.tensor(
        [
            [focal_length, 0, c_x],
            [0, focal_length, c_y],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    return K
