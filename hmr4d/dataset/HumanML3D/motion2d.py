import os
from pathlib import Path
from hmr4d.utils.pylogger import Log
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

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.dataset.HumanML3D.utils import load_motion_files, swap_left_right, upsample_motion
from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch, normalize_kp_2d
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
from hmr4d.utils.o3d_utils import o3d_skeleton_animation
from hmr4d.utils.plt_utils import plt_skeleton_animation
from hmr4d.utils.hml3d.utils import standardize_motion


SCALE = 1.0


# For exporting joints3d.pth
class BaseDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/amass/smplhg_raw",
        split="train",  # No use here
        index_path="./inputs/hml3d/index.csv",
        humanact_path="./inputs/humanact12",
        target_fps=20,
        limit_size=None,
        export_data=False,
        is_randomtest=False,
        multiple_sample=1,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.index_path = index_path
        self.humanact_path = humanact_path
        self.is_randomtest = is_randomtest
        self.multiple_sample = multiple_sample
        Log.info(f"Loading HML3D {split}...")

        self._load_dataset()
        # Options
        # HumanML3D uses 20 fps
        self.target_fps = target_fps
        self.limit_size = limit_size  # common usage: making validation faster

        if export_data:
            self._build_body_models()  # NOTE: this is only necessary for building joints3d.pth

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.motion_files))
        # Original and mirrored
        return len(self.motion_files) * 2

    def _build_body_models(self):
        body_models = {
            "male": make_smplx("rich-smplh", gender="male"),
            "female": make_smplx("rich-smplh", gender="female"),
        }
        self.smpl = body_models

    def _load_dataset(self):
        self.motion_files, self.new_names, self.start_frames, self.end_frames = load_motion_files(
            self.root, self.humanact_path, self.index_path
        )

    def _load_data(self, idx):
        motion_files = self.motion_files[idx // 2]
        data = np.load(motion_files)
        return data

    def _process_data(self, data, idx):
        name = self.motion_files[idx // 2]
        new_name = self.new_names[idx // 2]
        if isinstance(data, np.ndarray):
            # humanact12
            joints_pos = torch.tensor(data[...], dtype=torch.float32)
            joints_pos = joints_pos[:, :22, :]
        else:
            # amass
            joints_pos = self._process_amass(data, idx // 2)

        if idx % 2 == 1:
            # mirror
            joints_pos = swap_left_right(joints_pos)

        # ay to ayfz
        T_ay2ayfz = compute_T_ayfz2ay(joints_pos[:1], inverse=True)[0]  # (4, 4)
        joints_pos = apply_T_on_points(joints_pos, T_ay2ayfz)
        length = joints_pos.shape[0]

        # Return
        return_data = {
            "length": length,  # Value = F
            "gt_motion": joints_pos.float(),  # (F, 22, 3)
            "name": name,
            "new_name": new_name,
            "is_mirror": idx % 2 == 1,
        }

        return return_data

    def _process_amass(self, data, idx):
        fps = data["mocap_framerate"].item()
        interval = int(fps) // self.target_fps
        gender = data["gender"].item()
        if isinstance(gender, bytes):
            # Exists some b'female'
            gender = gender.decode()
        betas = data["betas"][:10]
        poses = data["poses"][::interval]
        trans = data["trans"][::interval]

        betas = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
        poses = torch.tensor(poses, dtype=torch.float32)
        trans = torch.tensor(trans, dtype=torch.float32)

        # TODO: HumanML3D uses DMPL, do not check very carefully

        output = self.smpl[gender](
            betas=betas.repeat(poses.shape[0], 1), body_pose=poses[:, 3:66], global_orient=poses[:, :3], transl=trans
        )
        joints_pos = output.joints[:, :22]  # T, J, 3

        # humanml3d pipeline
        ###############################################################
        trans_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        trans_matrix = torch.tensor(trans_matrix, dtype=torch.float32)
        joints_pos = matrix.get_position_from_rotmat(joints_pos, trans_matrix[None])

        source_path = self.motion_files[idx]
        start_frame = self.start_frames[idx]
        end_frame = self.end_frames[idx]

        if "Eyes_Japan_Dataset" in source_path:
            joints_pos = joints_pos[3 * self.target_fps :]
        if "MPI_HDM05" in source_path:
            joints_pos = joints_pos[3 * self.target_fps :]
        if "TotalCapture" in source_path:
            joints_pos = joints_pos[1 * self.target_fps :]
        if "MPI_Limits" in source_path:
            joints_pos = joints_pos[1 * self.target_fps :]
        if "Transitions_mocap" in source_path:
            joints_pos = joints_pos[int(0.5 * self.target_fps) :]
        joints_pos = joints_pos[start_frame:end_frame]

        joints_pos[..., 0] *= -1
        ################################################################

        return joints_pos

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


# for training
class Dataset(BaseDataset):
    def __init__(
        self,
        min_motion_time=2,
        max_motion_time=10,
        max_text_len=20,
        is_ignore_transl=True,
        unit_length=4,
        is_root_next=False,
        is_pinhole=False,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        is_notext=False,
        eleva_angle=None,
        train_fps=20,
        distance=1.0,
        patch_size=224,
        **kwargs,
    ):
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_root_next = is_root_next
        self.is_pinhole = is_pinhole
        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text

        self.is_notext = is_notext
        self.eleva_angle = eleva_angle
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size
        super().__init__(**kwargs)
        if self.is_notext:
            Log.info(f"Do not use text input!")
        else:
            Log.info(f"Required text: {self.required_text}; Anti text: {self.anti_text}")
        if self.is_pinhole:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
            Log.info(f"Use K with patch_size={self.patch_size} and distance={self.distance}")
            Log.info(f"Use eleva_angle={self.eleva_angle}")
        else:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
            Log.info(f"Use K with patch_size={self.patch_size} and distance={self.distance} but for orthographic")
            Log.info(f"Use eleva_angle={self.eleva_angle}")
            
        Log.info(f"DATASET INIT: HumanML3D dataset")
        self.default_text = ["good", "best", "high quality", "realistic", "natural", "smooth", "highly detailed"]
        self.default_text_threshold = 0.0


    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = torch.load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
        # Dict of {"joints3d": tensor(F, J, C), "name": str}
        self.idx2meta = self.prepare_meta(self.split)
        if self.multiple_sample > 1:
            self.idx2meta = self.idx2meta * self.multiple_sample
            Log.info(f"Enable multiple_sample={self.multiple_sample}")
        Log.info(f"Loaded {len(self.idx2meta)} data from {self.root}")
        f_num = 0
        for k in self.motion_files.keys():
            f_num += self.motion_files[k]["joints3d"].shape[0]
        hours = f_num / 20.0 / 3600.0 / 2  # there is mirror data
        Log.info(f"[{self.root}] [{self.split}] has {hours:.1f} hours motion.")

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start_frame, end_frame, text_list = meta
        seq_name_ = seq_name + ".npy"
        joints3d = self.motion_files[seq_name_]["joints3d"]
        joints3d = joints3d[start_frame:end_frame]
        return_data = {"joints3d": joints3d, "text_list": text_list}
        return return_data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)
    
    def _augment_text(self, ori_text):
        if torch.rand(1) < self.default_text_threshold and self.split != "test":
            selected_n = torch.randint(low=1, high=len(self.default_text), size=(1,)).item()
            selected_ind = torch.randperm(len(self.default_text))[:selected_n]
            selected_text = [self.default_text[i] for i in selected_ind]
            text = ". ".join(selected_text)
            text += "."
            text = " " + text
        else:
            text = ""
        return ori_text + text

    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        J = joints_pos.shape[1]
        J_ori = ori_joints_pos.shape[1]
        length = joints_pos.shape[0]

        distance = torch.ones((1,)) * self.distance
        angle = torch.rand((1,)) * 2 * torch.pi
        # [-30, 30] eleva
        if self.eleva_angle is not None:
            eleva_angle = torch.ones((1,)) * self.eleva_angle / 180.0 * torch.pi
        else:
            eleva_angle = (torch.rand((1,)) * 2 - 1) * 30.0 / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # 1, 4, 4
        T_w2c = torch.inverse(cam_mat)[0]  # 4, 4
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # F, J, 3
        i_motion2d = project_p2d(c_motion, self.K[None], is_pinhole=self.is_pinhole)  # (F, J, 2)
        c_motion2d = cvt_p2d_from_i_to_c(i_motion2d, self.K[None])  # (F, J, 2)
        if self.is_pinhole:
            normed_motion2d = normalize_keypoints_to_patch(i_motion2d, crop_size=self.patch_size)
            scale = 1.0
            # # this is different, the difference is upper normed_motion2d = lower normed_motion2d * scale * 200 / 224
            # on projected 3D, it is usually < 1, around 0.8
            normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d, not_moving=True) 
            scale = bbox[0, -1]
        else:
            normed_motion2d = i_motion2d
            # normed_motion2d = normalize_keypoints_to_patch(i_motion2d, crop_size=self.patch_size)
            normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d, not_moving=True) 
            # exactly same
            # normed_motion2d2, bbox_motion2d, bbox = normalize_kp_2d(c_motion2d, not_moving=True) 
            scale = 1.0

        # o3d_skeleton_animation(
        #     joints_pos, pos_2d=c_motion2d[None], w2c=T_w2c[None], is_pinhole=self.is_pinhole, name=caption
        # )
        # plt_skeleton_animation(normed_motion2d, skeleton_type="smpl")

        # remove root as it is always at (0, 0)
        normed_motion2d = normed_motion2d[:, 1:]
        J_2D = normed_motion2d.shape[1]

        max_motion_len = self.max_motion_time * self.train_fps

        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            normed_motion2d = torch.cat([normed_motion2d, torch.zeros((pad_length, J_2D, 2))], dim=0)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        # DEBUG
        # i_p2d = cvt_from_bi01_p2d(bi01_motion2d, bbx_lurb[None])  # (F, J, 2)
        # c_p2d = cvt_p2d_from_i_to_c(i_p2d[None], self.K[None])  # (1, F, J, 2)
        # o3d_skeleton_animation(joints_pos[None], c_p2d.reshape(1, 1, -1, 2), T_w2c[None], self.is_pinhole, caption)
        pred_masks = torch.ones((max_motion_len, J_2D), dtype=torch.float32)

        scale = torch.tensor([scale], dtype=torch.float32) 
        cam_emb = torch.cat([distance, torch.sin(eleva_angle), torch.cos(eleva_angle), scale], dim=-1)

        caption = self._augment_text(caption)

        # Return
        #Log.info(f"hml pred_masks: {pred_masks.shape}")
        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  # (F, 22 - 1, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "mask": pred_masks.float(), # (F, 22)
            "cam_emb": cam_emb.float(),  # (4,)
            # "word_embeddings": word_embeddings,
            # "pos_one_hots": pos_one_hots,
            "text": caption,
            "task": "2D",
        }
        return return_data

    def _process_motion(self, joints_pos):
        if self.train_fps != 20:
            joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        if self.is_root_next:
            # Add root velocity
            root_next = joints_pos[1:, :1]
            root_next = torch.cat([root_next, root_next[-1:]], dim=0)  # F, 1, C
            joints_pos = torch.cat([joints_pos, root_next], dim=1)  # F, J + 1, C

        # MDM does not move the character to the first frame when sample with another start_frame
        if self.is_ignore_transl:
            # move the every frame pelvis to origin
            joints_pos = joints_pos - joints_pos[:, :1, :]  # F, J, C
        else:
            # move the first frame pelvis to origin
            joints_pos = joints_pos - joints_pos[0, :1, :]  # F, J, C

        # # Crop the motions in to times of 4, and introduce small variations
        # if self.unit_length < 10:
        #     coin2 = np.random.choice(["single", "single", "double"])
        # else:
        #     coin2 = "single"

        # m_length = joints_pos.shape[0]
        # if coin2 == "double":
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == "single":
        #     m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(joints_pos) - m_length)
        # joints_pos = joints_pos[idx : idx + m_length]
        # ori_joints_pos = ori_joints_pos[idx : idx + m_length]
        return joints_pos, ori_joints_pos

    def prepare_meta(self, split):
        meta = []
        split_file = f"./inputs/hml3d/{split}.txt"
        txt_path = f"./inputs/hml3d/texts"

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

        max_motion_len = self.max_motion_time * 20
        min_motion_len = self.min_motion_time * 20

        # https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/data/dataset.py#L225
        # Original hml vec has F - 1 frames, so slightly different number of data.
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                seq_name = line.strip()
                if seq_name + ".npy" in self.motion_files.keys():
                    motion = self.motion_files[seq_name + ".npy"]["joints3d"]
                    motion_len = motion.shape[0]
                    # Follow MDM, only uses [2s ~ 10s]
                    if motion_len < min_motion_len or motion_len > max_motion_len:
                        continue
                    text_data = []
                    flag = False
                    with cs.open(os.path.join(txt_path, seq_name + ".txt")) as text_f:
                        for text_line in text_f.readlines():
                            text_dict = {}
                            line_split = text_line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    flag = True
                                    text_data.append(text_dict)
                            else:
                                start_frame = int(f_tag * 20)
                                end_frame = int(to_tag * 20)
                                n_motion = motion[start_frame:end_frame]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) > max_motion_len):
                                    continue
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    meta.append([seq_name, start_frame, end_frame, [text_dict]])
                    if flag:
                        meta.append([seq_name, 0, motion_len, text_data])
        return meta
#"""
class MVDataset(Dataset):
    def __init__(
        self,
        is_uniform_views=False,
        max_angle=None,
        N_views=5,
        is_mm=False,
        **kwargs,
    ):
        ####### Very strange, as SMPLER-X uses focal=5000 ##########
        # THUV2 mean K, distance=40
        # self.K = torch.tensor(
        #     [
        #         [4.5667e03, 0.0000e00, 1.1134e02],
        #         [0.0000e00, 4.5667e03, 1.1781e02],
        #         [0.0000e00, 0.0000e00, 1.0000e00],
        #     ]
        # )
        #############################################################
        #Log.info(f"MVDATASET INIT Before super().__init__: HumanML3D dataset")
        #super().__init__(**kwargs)
        self.is_uniform_views = is_uniform_views
        self.max_angle = max_angle  # half of the maxium sampling range, which is [-max_angle, max_angle]
        self.N_views = N_views
        Log.info(f"N_views={self.N_views}, max_angle={self.max_angle}, is_uniform={self.is_uniform_views}")
        max_angle_ = self.max_angle if self.max_angle is not None else 180
        Log.info(f"Interval: {2.0 * max_angle_ / self.N_views}")
        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        if is_mm:
            self.mm_idx = np.random.permutation(len(self.idx2meta))
        # Zehong advice, patch_size=224, distance = 4.5, FoV = 55
        super().__init__(**kwargs)

    def _process_data(self, data, idx):
        
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption

        max_motion_len = self.max_motion_time * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        F, J = joints_pos.shape[:2]

        if F > max_motion_len:
            start = torch.randint(0, F - max_motion_len, (1,)).item()
            end = start + max_motion_len
            joints_pos = joints_pos[start:end]
            F = joints_pos.shape[0]

        text_list = data["text_list"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        #caption = "" if self.is_notext else caption

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        length = joints_pos.shape[0]

        F, J, _ = joints_pos.shape
        _, J_ori, _ = ori_joints_pos.shape

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        max_angle = self.max_angle / 180 * torch.pi if self.max_angle is not None else torch.pi
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * max_angle / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * max_angle
        eleva_angle = torch.ones((N_views,)) * self.eleva_angle / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4
        joints_pos = joints_pos.reshape(1, F * J, 3)  # 1, F*J, 3
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # N, FJ, 3

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        i_motion2d = project_p2d(c_motion, Ks, is_pinhole=self.is_pinhole)  # (N, F*J, 2)
        bbx_lurb = torch.tensor([0, 0, 1, 1], dtype=torch.float32)  # 4
        bbx_lurb = bbx_lurb.reshape(1, 4).repeat(N_views, 1)  # N, 4
        bi01_motion2d = cvt_to_bi01_p2d(i_motion2d, bbx_lurb)  # (N, F*J, 2)
        bi01_motion2d = bi01_motion2d.reshape(N_views, F, J, 2)  # N, F, J, 2
        joints_pos = joints_pos.reshape(F, J, 3)  # F, J, 3

        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            bi01_motion2d = torch.cat([bi01_motion2d, torch.zeros((N_views, pad_length, J, 2))], dim=1)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        # DEBUG
        # i_p2d = cvt_from_bi01_p2d(bi01_motion2d, bbx_lurb[None])  # (F, J, 2)
        # c_p2d = cvt_p2d_from_i_to_c(i_p2d[None], self.K[None])  # (1, F, J, 2)
        # o3d_skeleton_animation(joints_pos[None], c_p2d.reshape(1, 1, -1, 2), T_w2c[None], self.is_pinhole, caption)

        scale = SCALE
        scale = 1.0
        scale = torch.tensor([scale], dtype=torch.float32)
        cam_emb = torch.cat([distance[:1], torch.sin(eleva_angle[:1]), torch.cos(eleva_angle[:1]), scale], dim=-1)

        # caption = "a person is playing the violin."
        # caption = "the person balances on one leg, and then puts their leg down."

        # Return
        return_data = {
            "length": length,  # Value = F
            "bbx_lurb": bbx_lurb.float(),  # (N, 4)
            "gt_motion2d": bi01_motion2d.float(),  # (N, F, 22, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "cam_emb": cam_emb.float(),  # (4,)
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "word_embs": word_embeddings.astype(np.float32),
            "pos_onehot": pos_one_hots.astype(np.float32),
            "text": caption,
            "text_len": sent_len,
            # "tokens": "_".join(tokens),
            "task": "3D",
        }
        return return_data

    def __getitem__(self, idx):
        if self.is_randomtest:
            idx = random.randint(0, len(self.idx2meta))
        return super().__getitem__(idx)


# Multi-view dataset for training multi-view generation
class MVCamDataset(MVDataset):
    def __init__(
        self,
        is_cam_rel2human=False,
        extra_path=None,
        **kwargs,
    ):
        self.is_cam_rel2human = is_cam_rel2human
        self.extra_path = extra_path

        super().__init__(**kwargs)
        Log.info(f"MVCAMDATASET INIT: HumanML3D dataset")

    def _load_dataset(self):
        super()._load_dataset()
        if self.extra_path is not None:
            print(f"Load saved data from {self.extra_path} for test")
            all_pth = os.listdir(self.extra_path)
            all_pth = [p for p in all_pth if p.endswith(".pth")]
            all_pth = sorted(all_pth)  # ["00000.pth", ...]
            self.saved_pred = []
            for i in range(len(all_pth)):
                k = all_pth[i]
                self.saved_pred.append(torch.load(os.path.join(self.extra_path, k)))
        else:
            self.saved_pred = None
    

    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        # random.seed(idx)
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        length = joints_pos.shape[0]

        F, J, _ = joints_pos.shape
        _, J_ori, _ = ori_joints_pos.shape

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        max_angle = self.max_angle / 180 * torch.pi if self.max_angle is not None else torch.pi
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * max_angle / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            start = torch.rand((1,)) * 2 * torch.pi
            angle = torch.rand((N_views,)) * 2 * max_angle
            angle = angle + start

        # [-30, 30] eleva
        if self.eleva_angle is not None:
            eleva_angle = torch.ones((1,)) * self.eleva_angle / 180.0 * torch.pi
        else:
            eleva_angle = (torch.rand((1,)) * 2 - 1) * 30.0 / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4

        T_w2c = torch.inverse(cam_mat)  # N, 4, 4
        joints_pos = joints_pos.reshape(1, F * J, 3)  # 1, F*J, 3
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # N, FJ, 3

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        i_motion2d = project_p2d(c_motion, Ks, is_pinhole=self.is_pinhole)  # (N, F*J, 2)
        c_motion2d = cvt_p2d_from_i_to_c(i_motion2d, Ks)  # (N, F*J, 2)
        i_motion2d = i_motion2d.reshape(N_views, F, J, 2)
        c_motion2d = c_motion2d.reshape(N_views, F, J, 2)
        if self.is_pinhole:
            normed_motion2d = normalize_keypoints_to_patch(i_motion2d.clone(), crop_size=self.patch_size)
            normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d.clone(), not_moving=True, multiview=True)
            scale = bbox[0, -1]
            if self.split == "test":
                normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d.clone(), not_moving=True, multiview=True, randselect=False)
                scale = bbox[0, -1]
                scale = SCALE
        else:
            normed_motion2d = i_motion2d
            normed_motion2d, bbox_motion2d, bbox = normalize_kp_2d(i_motion2d.clone(), not_moving=True, multiview=True)
            scale = 1.0

        joints_pos = joints_pos.reshape(F, J, 3)  # F, J, 3

        # DEBUG
        # o3d_skeleton_animation(joints_pos, pos_2d=c_motion2d, w2c=T_w2c, is_pinhole=self.is_pinhole, name=caption)

        # remove zero root
        normed_motion2d = normed_motion2d[:, :, 1:]
        J_2D = normed_motion2d.shape[-2]

        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            normed_motion2d = torch.cat([normed_motion2d, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat))  # N, 3
        theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]

        if self.is_cam_rel2human:
            d_T = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=-1)  # N, 4
        else:
            # delta = target - condition
            d_theta = theta - theta[:1]  # N, 1
            d_azimuth = (azimuth - azimuth[:1]) % (2 * torch.pi)  # N, 1
            d_z = z - z[:1]  # N, 1
            d_T = torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1)  # N, 4

        scale = torch.tensor([[scale]], dtype=torch.float32) 
        d_T = torch.cat([d_T, scale.repeat(N_views, 1)], dim=-1)  # N, 5

        cam_emb_2d = torch.cat([distance[:1], torch.sin(eleva_angle), torch.cos(eleva_angle), scale[0]], dim=-1)

        # caption = "a person is playing the violin."
        # caption = "a person plays the violin."
        # caption ="C plays the violin."
        # caption = "the person balances on one leg, and then puts their leg down."

        caption = self._augment_text(caption)
        
        if self.saved_pred is not None:
            pred_joints = self.saved_pred[idx]["pred"] # (F, 22, 3)
        else:
            pred_joints = None


        # Return
        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  # (N, F, 22, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "cam_emb": d_T.float(),  # N, 5
            "2d_cam_emb": cam_emb_2d.float(),  # (4,)
            "text": caption,
            "task": "3D",
        }

        if self.split == "test":
            test_dict = {
                "word_embs": word_embeddings.astype(np.float32),
                "pos_onehot": pos_one_hots.astype(np.float32),
                "text_len": sent_len,
                "tokens": "_".join(tokens),
            }
            return_data.update(test_dict)
            if pred_joints is not None:
                return_data["pred_joints"] = pred_joints
        return return_data

class NewTextMVDataset(MVDataset):
    def _load_dataset(self):
        self.text_files = [
            # OMG
            "Play basketball",
            "Air squat workout",
            "Pitching a baseball.",
            "Moves forward with arms moving dancing and then a turn then walks back.",
            "The person is flying like a airplane.",
            "A person bends their back to stretch.",
            "Dance ballet jazz.",
            "A powerful man unleashes strikes with his fists while dodging.",
            "A graceful dancer moves in ballet steps with rhythm.",
            "A person spinkick with taekwondo skills.",
            "A person breakdance footwork.",
            "A person throw shuriken while leaping back.",
            "A person kneeling warming hands over campfire.",
            ########
            "A person is doing push up.",
            "A person is climbing up.",
            "A person is lying down.",
            "A person is turning around.",
            "A person is punching someone.",
            "A person is boxing.",
            "A person is striking someone.",
            "A person is hitting someone.",
            "A person plays kung fu.",
            "A person is playing kung fu.",
            "A person is cartwheeling.",
            "A person is boating.",
            "A person is going down on his knees.",
            "A person jumps high to touch something.",
            "A person prays.",
            "A person turns around and sits down.",
            "A person throws a ball with two hands.",
            "A person plays basketball.",
            "A person kicks left leg.",
            "A person touches his left foot with right hand.",
            "A person plays basketball.",
            "A person plays soccer.",
            "A person sits down.",
            "A person is lifting up a box.",
            "A person is carrying a box.",
            "A person is sitting on a chair.",
            "A person is sitting.",
            "A person is walking.",
            "A person is running.",
            "A person is jumping.",
            "A person is dancing.",
            "A person is drumming.",
            "A person is playing the piano.",
            "A person is playing the drums.",
            "A person is playing the violin.",
            "A person is playing the guitar.",
        ]
        
        # txt_path = "inputs/ezi_data/Motion_Xpp_v0.json"
        self.N = 1

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.text_files))
        return len(self.text_files) * self.N

    def _load_data(self, idx):
        return self.text_files[idx // self.N]

    def _process_data(self, data, idx):
        caption = data
        caption = "" if self.is_notext else caption
        length = torch.randint((self.max_motion_time - 2) * 30, self.max_motion_time * 30, (1,)).item()
        length = np.random.randint((self.max_motion_time - 4) * 30, (self.max_motion_time - 2) * 30)
        length = 150 # test speed

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        max_angle = self.max_angle / 180 * torch.pi if self.max_angle is not None else torch.pi
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * max_angle / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * max_angle
        eleva_angle = torch.ones((N_views,)) * self.eleva_angle / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        
        scale = SCALE
        scale = torch.tensor([scale], dtype=torch.float32)
        cam_emb = torch.cat([distance[:1], torch.sin(eleva_angle[:1]), torch.cos(eleva_angle[:1]), scale], dim=-1)

        # Return
        return_data = {
            "length": length,  # Value = F
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "cam_emb": cam_emb.float(),  # (3,)
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "text": caption,
            "task": "3D",
        }
        return return_data


class NewTextDataset(NewTextMVDataset):
    def _process_data(self, data, idx):
        caption = data
        caption = "" if self.is_notext else caption
        length = torch.randint((self.max_motion_time - 2) * 30, self.max_motion_time * 30, (1,)).item()
        length = np.random.randint((self.max_motion_time - 6) * 30, (self.max_motion_time - 4) * 30)

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        max_angle = self.max_angle / 180 * torch.pi if self.max_angle is not None else torch.pi
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * max_angle / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * max_angle
        eleva_angle = torch.ones((N_views,)) * self.eleva_angle / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        
        scale = SCALE
        scale = torch.tensor([scale], dtype=torch.float32)
        cam_emb = torch.cat([distance[:1], torch.sin(eleva_angle[:1]), torch.cos(eleva_angle[:1]), scale], dim=-1)

        # Return
        return_data = {
            "length": length,  # Value = F
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "cam_emb": cam_emb.float(),  # (3,)
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "text": caption,
            "task": "2D",
        }
        return return_data


class NewTextMVCamDataset(NewTextMVDataset):
    def _process_data(self, data, idx):
        caption = data
        caption = "" if self.is_notext else caption
        if "soccer" in caption:
            length = np.random.randint((self.max_motion_time - 4) * 30, (self.max_motion_time - 2) * 30)
        elif "danc" in caption:
            length = np.random.randint((self.max_motion_time - 3) * 30, (self.max_motion_time - 1) * 30)
        else:
            length = np.random.randint((self.max_motion_time - 6) * 30, (self.max_motion_time - 4) * 30)
        length = 150 # test speed

        N_views = self.N_views

        distance = torch.ones((N_views,)) * self.distance
        max_angle = self.max_angle / 180 * torch.pi if self.max_angle is not None else torch.pi
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * max_angle / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            start = torch.rand((1,)) * 2 * torch.pi
            angle = torch.rand((N_views,)) * 2 * max_angle
            angle = angle + start

        # [-30, 30] eleva
        if self.eleva_angle is not None:
            eleva_angle = torch.ones((1,)) * self.eleva_angle / 180.0 * torch.pi
        else:
            eleva_angle = (torch.rand((1,)) * 2 - 1) * 30.0 / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4

        T_w2c = torch.inverse(cam_mat)  # N, 4, 4

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3

        spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat))  # N, 3
        theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]

        
        # delta = target - condition
        d_theta = theta - theta[:1]  # N, 1
        d_azimuth = (azimuth - azimuth[:1]) % (2 * torch.pi)  # N, 1
        d_z = z - z[:1]  # N, 1
        d_T = torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1)  # N, 4

        scale = SCALE
        scale = torch.tensor([[scale]], dtype=torch.float32) 
        d_T = torch.cat([d_T, scale.repeat(N_views, 1)], dim=-1)  # N, 5

        cam_emb_2d = torch.cat([distance[:1], torch.sin(eleva_angle), torch.cos(eleva_angle), scale[0]], dim=-1)

        # Return
        return_data = {
            "length": length,  # Value = F
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "Ks": Ks.float(),  # N, 3, 3
            "patch_size": self.patch_size,  # value
            "cam_emb": d_T.float(),  # N, 4
            "2d_cam_emb": cam_emb_2d.float(),  # (3,)
            "text": caption,
            "task": "3D",
        }
        return return_data
