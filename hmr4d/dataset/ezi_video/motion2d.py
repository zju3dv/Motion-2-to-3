import os
from pathlib import Path
from hmr4d.utils.pylogger import Log
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import torch
import cv2
from torch.utils import data
import codecs as cs
import json
import decord
from decord import cpu, gpu
from hmr4d.utils.smplx_utils import make_smplx
import random
from copy import deepcopy
import re
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.dataset.motionx.utils import normalize_kp_2d, adjust_K, estimate_focal_length, generate_camera_intrinsics
from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch
from hmr4d.dataset.HumanML3D.utils import upsample_motion
from hmr4d.dataset.ezi_video.utils import load_motion_files, load_pkl, smpl_fk, augment_motion_files
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface, cartesian_to_spherical
from hmr4d.network.evaluator.word_vectorizer import WordVectorizer
from hmr4d.utils.video_io_utils import read_video_np
from hmr4d.utils.vis.renderer import Renderer
import imageio
from hmr4d.utils.o3d_utils import o3d_skeleton_animation, vis_smpl_forward_animation
from hmr4d.utils.plt_utils import plt_skeleton_animation
from hmr4d.utils.hml3d.utils import standardize_motion

# For exporting joints3d.pth
class BaseDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/ezi_video/music",
        split="train",  # No use here
        target_fps=30,
        limit_size=None,
        export_data=False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = split
        Log.info(f"Loading from {root} {split}...")

        self._load_dataset()
        # Options
        self.target_fps = target_fps
        self.limit_size = limit_size  # common usage: making validation faster

        self._build_body_models()
        self.is_egoexo = False
        if "egoexo" in root:
            self.is_egoexo = True

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.motion_files))
        # Original and mirrored
        return len(self.motion_files)

    def _build_body_models(self):
        self.smpl = None

    def _load_dataset(self):
        self.motion_files = load_motion_files(self.root)

    def _load_data(self, idx):
        motion_files = self.motion_files[idx]
        pkl_path = os.path.join(motion_files, "ezi_output_merged.pkl")
        data = load_pkl(pkl_path)
        return data

    def _process_data(self, data, idx):
        if data is None:
            return {"length": False}
        name = self.motion_files[idx]
        video_path = name.replace("idea400_result", "idea400_video") + ".mp4"
       

        #Directly get 2D body pose
        joints = torch.tensor(data["motion2d_points"], dtype=torch.float32)  # (F, 22, 2)
        pred_scores = torch.tensor(data["pred_scores"], dtype=torch.float32)  # F, 10
        bboxs = torch.tensor(data["bboxs"], dtype=torch.float32)
        #print(f"bboxs: {bboxs.shape}")
        F =  joints.shape[0]
        # vis_smpl_forward_animation(transl, torch.cat([global_orient, body_pose], dim=-1))
        # Return
        file_name = name.split("/")[-1]
        if self.is_egoexo:
            file_name = name.split("/")[-2] + "/" + file_name
        return_data = {
            "length": F,  # Value = F
            "bboxs": bboxs,  # (F, 4)
            "incam_joints": joints,  # (F, J, 2)
            "pred_scores": pred_scores,  # (F, J)
            "name": file_name,  # str
            "video_path": video_path,  # str
        }

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        #Log.info(f"Loaded {data['name']} with length {data['length']}, and keys: {data.keys()}")
        return data

# for training
class Dataset(BaseDataset):
    def __init__(
        self,
        min_motion_time=2,
        max_motion_time=10,
        train_fps=30,
        is_root_next=False,
        patch_size=224,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        threshold_2d=0.4,
        text_path="inputs/smplerx_data/idea400_halfway_4.14.v10.json",
        text_path2="inputs/smplerx_data/idea400_halfway_4.14.v10.json",
        **kwargs,
    ):

        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.train_fps = train_fps
        self.is_root_next = is_root_next
        self.patch_size = patch_size

        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text

        self.threshold_2d = threshold_2d
        Log.info(f"Use threshold_2d: {self.threshold_2d}")

        self.text_path = text_path
        self.text_path2 = text_path2
        super().__init__(**kwargs)
        Log.info(f"Required text: {self.required_text}; Anti text: {self.anti_text}")

    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = torch.load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
        self.is_egoexo = "egoexo" in self.root
        self.motion_files = augment_motion_files(self.motion_files, is_notegoexo=not self.is_egoexo)
        self.idx2meta, max_L = self.prepare_meta(self.text_path, self.text_path2, self.split)
        Log.info(f"Loaded {len(self.idx2meta)} data from {self.root} with max_L = {max_L}, txt_path: {self.text_path}")
        # self._print_score_distribution()
    
    def _print_score_distribution(self):
        all_ratio = []
        all_length = []
        for i in range(len(self.idx2meta)):
            data = self._load_data(i)
            pred_scores = data["pred_scores"]
            ratio = (pred_scores > self.threshold_2d).sum() / pred_scores.numel()
            if data["pred_scores"].shape[0] != data["incam_joints"].shape[0] and data["incam_joints"].shape[0] < 64:
                Log.info(f"pred_scores: {pred_scores.shape}, incam_joints: {data['incam_joints'].shape}")
            length = data["length"]
            all_length.append(length)
            all_ratio.append(ratio)
        all_ratio = np.array(ratio)
        Log.info(f"Larger than {self.threshold_2d} ratio: {np.mean(all_ratio)}")
        avg_length = np.array(all_length).mean()
        Log.info(f"Average length {avg_length}")
        return

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start_frame, end_frame, text1, text2 = meta
        return_data = {}
        nosubsample_keys = ["video_path", "name", "length"]
        for k, v in self.motion_files[seq_name].items():
            if k in nosubsample_keys:
                return_data[k] = v
            else:
                return_data[k] = v[start_frame:end_frame]
        if text2 == "":
            return_data["text"] = text1
        else:
            if torch.rand(1) > 0.7:
                return_data["text"] = text1
            else:
                return_data["text"] = text2

        if self.is_egoexo and "aug" not in seq_name and torch.rand(1) < 0.5:
            return_data = self._augment_online(return_data, seq_name)
        
        if self.is_egoexo and torch.rand(1) < 0.1:
            # seq_name: upenn_0713_Dance_5_6_0019/02 not exactly same
            # root: inputs/ezi_data/egoexo_basketball_split_v1.pth
            activity = self.root.split("/")[-1].split("_")[1]
            return_data["text"] = f"{activity}"

        return return_data

    def _augment_online(self, data, seq_name, max_merge_n=3):
        """在线合并函数"""
        combined_incam_joints = data["incam_joints"]
        combined_bboxs = data["bboxs"]
        combined_pred_scores = data["pred_scores"]
        combined_name = [data["name"]]
        combined_video_path = [data["video_path"]]

        frame_num = int(data["name"].split("_")[-1].split("/")[0])
        base_name, cam_id = seq_name.rsplit('_', 1)[0], data["name"].split('/')[-1]

        # 在线合并相邻段的逻辑
        for i in range(1, random.randint(2, max_merge_n)):
            next_frame_num = f"{frame_num + i:04d}"
            adjacent_key = f"{base_name}_{next_frame_num}/{cam_id}"

            # 检查相邻段是否存在
            if adjacent_key not in self.motion_files:
                break

            # 获取相邻段并合并
            seg = self.motion_files[adjacent_key]
            seg_incam_joints = seg["incam_joints"][:seg["pred_scores"].shape[0]]
            seg_bboxs = seg["bboxs"][:seg["pred_scores"].shape[0]]
            seg_pred_scores = seg["pred_scores"]

            combined_incam_joints = torch.cat([combined_incam_joints, seg_incam_joints], dim=0)
            combined_bboxs = torch.cat([combined_bboxs, seg_bboxs], dim=0)
            combined_pred_scores = torch.cat([combined_pred_scores, seg_pred_scores], dim=0)
            combined_name.append(seg["name"])
            combined_video_path.append(seg["video_path"])

            if combined_incam_joints.shape[0] > 300:
                break

        # 更新合并后的数据
        data["incam_joints"] = combined_incam_joints
        data["bboxs"] = combined_bboxs
        data["pred_scores"] = combined_pred_scores
        data["name"] = combined_name
        data["video_path"] = combined_video_path
        data["length"] = combined_incam_joints.shape[0]

        return data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        #Log.info("self.idx2meta:", len(self.idx2meta))
        return len(self.idx2meta)

    def _get_joints2d(self, joints_pos, pred_scores):
        i_motion2d_1 = joints_pos.clone().detach()
        if isinstance(pred_scores, np.ndarray):
            pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
        else:
            pred_scores = pred_scores.float().clone()
        not_moving = False
        if "music" in self.root:
            not_moving = True

        L  = i_motion2d_1.shape[0]
        if self.is_root_next:
                # Add next frame root
                root_next = i_motion2d_1[1:, :1]  # (F-1, 1, 2)
                root_next = torch.cat([root_next, root_next[-1:]], dim=0)  # F, 1, 2
                i_motion2d_1 = torch.cat([i_motion2d_1, root_next], dim=1)  # F, J + 1, 2
                root_next_score = pred_scores[1:, :1]  # (F-1, 1)
                root_next_score = torch.cat([root_next_score, root_next_score[-1:]], dim=0)  # F, 1
                pred_scores = torch.cat([pred_scores, root_next_score], dim=1)  # F, J + 1
        
        L = pred_scores.shape[0] # 1 - 0
        fail_mask = pred_scores < self.threshold_2d # (F, J)
        fail_count = fail_mask.sum(dim=0, keepdim=True) # (1, J)
        fail_joint_mask = fail_count > int(L * 0.5) # (1, J)
        new_score = pred_scores.clone()
        new_score = new_score * ~fail_joint_mask
        normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d_1, not_moving=not_moving)
        return normed_motion2d, new_score

    def _process_data(self, data, idx):
        #Log.info(f"{data.keys() }")
        joints_pos = data["incam_joints"] # 2D Joints  #(F, J, 2)
        pred_scores = data["pred_scores"] #(F, J)
        l = joints_pos.shape[0] # joints are smoothed (appended to 64 frames), so remove appending frames
        p_l = pred_scores.shape[0] # scores
        if p_l < l:
            joints_pos = joints_pos[:p_l]
        
        video_path = data["video_path"]
        if isinstance(video_path, list):
            video_path = video_path[0]

        data_fps = 15 if "15fps" in video_path else 30
        if self.train_fps != data_fps:
            joints_pos = upsample_motion(joints_pos, data_fps, self.train_fps)

        max_motion_len = self.max_motion_time * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        length, J = joints_pos.shape[:2]

        if length > max_motion_len:
            start = torch.randint(0, length - max_motion_len, (1,)).item()
            end = start + max_motion_len
            joints_pos = joints_pos[start:end]
            pred_scores  = pred_scores[start:end]
            length = joints_pos.shape[0]

        caption = deepcopy(data["text"])
        # Change the C to He or She or the person.
        if "egoexo" in self.root and "gpt" not in self.text_path:
            #Log.info(f"Name: {data['name']}")
            name_list = ["he ", "She ", "He ", "she ", "The person ", "the person ", "The woman ", "the woman ", "the man ", "The man "]
            name_list2 = ["to him", "to her", "to another person", "to the other person",  "to another person", "to the other person"]
            rand_num = torch.rand(1)

            rand_name_int2 = torch.randint(0, len(name_list2)-1, (1,)).item()
            rand_name2 = name_list2[rand_name_int2]
            if rand_num > 0.8:
                rand_name_int = torch.randint(0, len(name_list)-1, (1,)).item()
                rand_name = name_list[rand_name_int]
            elif rand_num > 0.4:
                if torch.rand(1) > 0.5:
                    rand_name = "the person "
                else:
                    rand_name = "a person "
            else:
                if torch.rand(1) > 0.5:
                    rand_name = "The person "
                else:
                    rand_name = "A person "
            caption = caption.replace("to C", rand_name2)
            caption = caption.replace("to D", rand_name2)
            caption = caption.replace("C ", rand_name)
            caption = caption.replace("D ", rand_name)
            caption = re.sub(r"A man [A-Z]", rand_name, caption)
            caption = re.sub(r"The man [A-Z]", rand_name, caption)
            caption = re.sub(r"a man [A-Z]", rand_name2, caption)
            caption = re.sub(r"the man [A-Z]", rand_name2, caption)
            caption = re.sub(r"a player [A-Z]", "a player", caption)
            caption = re.sub(r"A player [A-Z]", "A player", caption)
            caption = re.sub(r"the player [A-Z]", "the player", caption)
            caption = re.sub(r"The player [A-Z]", "The player", caption)
            caption = re.sub(r"the person [A-Z]", "the person", caption)
            caption = re.sub(r"The person [A-Z]", "The person", caption)
        # Random drop the last sentence
        caption2 = deepcopy(caption)
        split_text = caption2.split('.')
        
        if len(split_text) > 1 and torch.rand(1) > 0.5:
            rand_int = torch.randint(0, len(split_text)-1, (1,)).item()
            caption = split_text[rand_int] + "."

        normed_motion2d, pred_masks = self._get_joints2d(joints_pos, pred_scores=pred_scores)
        # plt_skeleton_animation(normed_motion2d, skeleton_type="smpl")
        # remove root as it is always at (0, 0)
        normed_motion2d = normed_motion2d[:, 1:]
        pred_masks = pred_masks[:,1:]
        #Log.info(f'get joints pred_masks 2: {pred_masks[0,:]}')
        J_2D = normed_motion2d.shape[1]
        #print(f"normed_motion2d: {normed_motion2d.shape}, pred_masks:  {pred_masks.shape}")
        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            #joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            normed_motion2d = torch.cat([normed_motion2d, torch.zeros((pad_length, J_2D, 2))], dim=0)
            pred_masks = torch.cat([pred_masks, torch.zeros((pad_length, J_2D))], dim=0)
        
        gt_motion = torch.zeros((max_motion_len, J, 3), dtype=torch.float32)

        cam_emb = torch.zeros(4, dtype=torch.float32)

        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  # (F, 22, 2)
            "gt_motion": gt_motion.float(),  # (F, 22, 3)
            "is_pinhole": True,  # Value = False or True
            "mask": pred_masks.float(),  # (F, 22)
            "cam_emb": cam_emb.float(),  # (4,)
            "text": caption,
            "task": "2D",
        }
        
        #for key in return_data.keys():
        #    if key == "mask":
        #        print(f"{key}: {return_data[key]}")
            
                
        return return_data

    def prepare_meta(self, txt_path, txt_path2, split):
        meta = []

        # Filter data by required_text
        def is_contain_required_text(text, required_text, anti_text):
            flag = False
            if required_text is None:
                flag = True
            else:
                for r_t in required_text:
                    if r_t in text:
                        flag = True
            if anti_text is not None:
                for a_t in anti_text:
                    if a_t in text:
                        flag = False
            return flag

        MAX_L = -1
        with open(txt_path, "r") as file:
            data = json.load(file)
        with open(txt_path2, "r") as file:
            data2 = json.load(file)
        Log.info(f"Loaded {len(data)} data from {txt_path}")
        Log.info(f"Loaded {len(data)} data2 from {txt_path2}")

        for k in self.motion_files.keys():
            L = self.motion_files[k]["length"]
            # fitler out too short motions
            if L < 30:
                continue
            k = k.replace("perform500", "perform")
            name = self.motion_files[k]["name"]
                
            if  "egoexo" in txt_path: 
                if isinstance(name, list):
                   # 如果段数超过2，随机选择2个段进行合并
                    if len(name) > 2:
                        ind = torch.randint(0, len(name) - 1, (1,)).item()
                        name_ = []
                        for i in range(ind, ind + 2):
                            name_.append(name[i])
                        name = name_

                    captions1, captions2 = [], []
                    for n in name:
                        # 解析段名，提取路径信息
                        segment_key = "takes/" + n.split("/")[-2]
                        caption1_part = data.get(segment_key)
                        caption2_part = data2.get(segment_key)

                        # 根据找到的情况更新 captions
                        if caption1_part and not caption2_part:
                            caption2_part = caption1_part
                        elif caption2_part and not caption1_part:
                            caption1_part = caption2_part

                        # 如果两个都没有找到，则只添加空字符串
                        if caption1_part or caption2_part:
                            captions1.append(caption1_part or "")
                            captions2.append(caption2_part or "")
                        else:
                            captions1.append("")
                            captions2.append("")

                    # 合并captions
                    caption1 = " ".join(captions1)
                    caption2 = " ".join(captions2)

                elif  '/' in name and "takes/" + name.split("/")[-2] in data.keys():
                    caption1 = data["takes/" +name.split("/")[-2]]
                    if "takes/" + name.split("/")[-2] in data2.keys():
                        caption2 = data2["takes/" +name.split("/")[-2]]
                    else:
                        caption2 = caption1
                    
                else:
                    continue
            else:
                if name in data.keys():
                    caption1 = data[name]
                    if name in data2.keys():
                        caption2 = data2[name]
                    else:
                        caption2 = caption1
                else:
                    continue
            L = self.motion_files[k]["length"] # Change to 2D length
            MAX_L = max(MAX_L, L)
            if "Motion_Xpp" in txt_path:
                meta.append([k, 0, L, caption1, caption2])
            if "climb" in name:
                meta.append([k, 0, L, caption1, caption2])
                meta.append([k, 0, L, caption1, caption2])
            if "soccer" in name or "basketball" in name or "cpr" in name:
                meta.append([k, 0, L, caption1, caption2])
            meta.append([k, 0, L, caption1, caption2])

        return meta, MAX_L


# Multi-view dataset for MAS inference
class MVDataset(Dataset):
    def __init__(
        self,
        is_uniform_views=False,
        N_views=4,
        eleva_angle=5,
        distance=4.5,
        patch_size=224,
        is_notext=False,
        is_mm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_uniform_views = is_uniform_views
        self.N_views = N_views
        self.eleva_angle = eleva_angle
        self.distance = distance
        self.is_notext = is_notext
        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        if is_mm:
            self.mm_idx = np.random.permutation(len(self.idx2meta))
        self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)

    def _process_data(self, data, idx):
        #joints_pos = self._get_joints3d(data)
        joints_pos = data["incam_joints"]  # F, J, 2
        #cam_K = data["cam_K"]  # F, 4
        data_fps = 15 if "15fps" in data["video_path"] else 30
        #if self.train_fps != data_fps:
        #    joints_pos = upsample_motion(joints_pos, data_fps, self.train_fps)

        max_motion_len = self.max_motion_time * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        F, J = joints_pos.shape[:2]

        if F > max_motion_len:
            start = torch.randint(0, F - max_motion_len, (1,)).item()
            end = start + max_motion_len
            joints_pos = joints_pos[start:end]
            F = joints_pos.shape[0]
        caption = "" if self.is_notext else caption

        F, J, _ = joints_pos.shape

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * torch.pi / N_views /2.0  #*2.0/3.0
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * torch.pi

        eleva_angle = torch.ones((N_views,)) * self.eleva_angle / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3

        max_motion_len = self.max_motion_time * self.train_fps
        if F < max_motion_len:
            # pad
            pad_length = max_motion_len - F
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 2))], dim=0)
        joints_pos_test = torch.tensor(np.zeros((max_motion_len, J, 2))).clone().detach()
        # Return
        return_data = {
            "length": F,  # Value = F
            "gt_motion":  joints_pos_test.float(),#joints_pos.float(),  # (F, 22, 2) #Using 2D GT Motion
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": True,  # Value = False or True
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "2d_cam_emb": torch.zeros(4, dtype=torch.float32),
            "text": caption,
            "task": "3D",
        }
        return return_data
