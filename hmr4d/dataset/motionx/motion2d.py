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
from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch, normalize_kp_2d, load_motion_and_text, load_mixed
from hmr4d.dataset.wham_video.utils import smpl_fk
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
import spacy


SCALE = 1.0


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/motionx/motion_data/smplx_322",
        text="inputs/motionx/motionx_seq_text_v1.1",
        # subset="HAA500",
        # subset="game_motion",
        # subset="idea400",
        # subset="kungfu",
        subset="mixed",
        split="train",  # No use here
        limit_size=None,
        **kwargs,
    ):
        super().__init__()
        self.subset = subset
        if subset == "mixed":
            self.root = root
            self.text = text
        else:
            self.root = os.path.join(root, subset)
            self.text = os.path.join(text, subset)
        self.split = split
        Log.info(f"Loading from {self.root}...")

        self._load_dataset()
        # Options
        self.limit_size = limit_size  # common usage: making validation faster

        self._build_body_models()  # NOTE: this is only necessary for building joints3d.pth

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.motion_files))
        return len(self.motion_files)
    
    def _load_dataset(self):
        pass

    def _build_body_models(self):
        body_models = {
            "male": make_smplx("rich-smplx", gender="male"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
            "female": make_smplx("rich-smplx", gender="female"),
        }
        self.smpl = body_models
    
    def _load_data(self, idx):
        pass

    def _process_data(self, data, idx):
        pass
    
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
        is_root_next=False,
        is_pinhole=False,
        eleva_angle=None,
        distance=1.0,
        patch_size=224,
        **kwargs,
    ):
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.is_root_next = is_root_next
        self.is_pinhole = is_pinhole

        self.eleva_angle = eleva_angle
        self.distance = distance
        self.patch_size = patch_size

        self.token_model = spacy.load("en_core_web_sm")

        super().__init__(**kwargs)
        if self.is_pinhole:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
            Log.info(f"Use K with patch_size={self.patch_size} and distance={self.distance}")
            Log.info(f"Use eleva_angle={self.eleva_angle}")
        else:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
            Log.info(f"Use K with patch_size={self.patch_size} and distance={self.distance} but for orthographic")
            Log.info(f"Use eleva_angle={self.eleva_angle}")
            
    def _load_dataset(self):
        if self.subset == "mixed":
            self.motion_files = load_mixed(self.root, self.text)
            self.idx2meta = self.prepare_meta(self.split)
        else:
            self.motion_files = load_motion_and_text(self.root, self.text)
            # Dict of {"motion": tensor(F, J, C), "name": str}
            self.idx2meta = self.prepare_meta(self.split)
        f_num = 0
        for k in self.motion_files.keys():
            f_num += self.motion_files[k]["motion"].shape[0]
        hours = f_num / 20.0 / 3600.0 / 2  # there is mirror data
        Log.info(f"[{self.root}] has {hours:.1f} hours motion.")

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start_frame, end_frame, text_list = meta
        motion = self.motion_files[seq_name]["motion"]
        motion = motion[start_frame:end_frame]
        motion = torch.tensor(motion, dtype=torch.float32)
        smpl_params = {"global_orient": motion[:, :3],
                  "body_pose": motion[:, 3:66],
                  "transl": motion[:, 309:312],
                  "betas": None,
                  }

        joints = smpl_fk(self.smpl["neutral"], **smpl_params)  # (F, 22, 3)
        return_data = {"joints3d": joints, "text_list": text_list}
        return return_data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        caption = random.choice(text_list)

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

        max_motion_len = self.max_motion_time * 30

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
        return joints_pos, ori_joints_pos

    def prepare_meta(self, split):
        meta = []
        if split == "test" and self.subset != "mixed":
            idea400_text_path = f"./inputs/motionx/motionx_seq_text_v1.1/{self.subset}_test_seq_names.json"
            with open(idea400_text_path, "r") as file:
                test_seq_names = json.load(file)
            for k in test_seq_names:
                motion = self.motion_files[k]["motion"]
                text = self.motion_files[k]["text"]
                L = motion.shape[0]
                if L < self.min_motion_time * 30:
                    continue
                if L > self.max_motion_time * 30:
                    continue
                meta.append([k, 0, L, [text]])
            
        else:
            for k in self.motion_files.keys():
                motion = self.motion_files[k]["motion"]
                text = self.motion_files[k]["text"]
                L = motion.shape[0]
                if L < self.min_motion_time * 30:
                    continue
                if L > self.max_motion_time * 30:
                    continue
                meta.append([k, 0, L, [text]])
        
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
        caption = random.choice(text_list)
        caption = caption.replace('/', ' ')
        tokens = self.token_model(caption)
        token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
        tokens = token_format.split(" ")

        max_motion_len = self.max_motion_time * 30
        min_motion_len = self.min_motion_time * 30
        F, J = joints_pos.shape[:2]

        if F > max_motion_len:
            start = torch.randint(0, F - max_motion_len, (1,)).item()
            end = start + max_motion_len
            joints_pos = joints_pos[start:end]
            F = joints_pos.shape[0]

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

        max_motion_len = self.max_motion_time * 30
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
    
    def _load_dataset(self):
        super()._load_dataset()
        if self.extra_path is not None:
            print(f"Load saved data from {self.extra_path} for test")
            all_pth = os.listdir(self.extra_path)
            all_pth = [p for p in all_pth if p.endswith(".pth")]
            all_pth = sorted(all_pth)  # ["00000.pth", ...]
            self.saved_pred = []
            new_idx2meta = []
            for i in range(len(all_pth)):
                if i < 000 or i >= 800:
                    continue
                k = all_pth[i]
                self.saved_pred.append(torch.load(os.path.join(self.extra_path, k)))
                new_idx2meta.append(self.idx2meta[i])
            self.idx2meta = new_idx2meta
        else:
            self.saved_pred = None
    
    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        # random.seed(idx)
        caption = random.choice(text_list)
        caption = caption.replace('/', ' ')
        tokens = self.token_model(caption)
        token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
        tokens = token_format.split(" ")

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

        max_motion_len = self.max_motion_time * 30
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
        if self.saved_pred is not None:
            pred_joints = self.saved_pred[idx]["pred"] # (F, 22, 3)
            max_motion_len = self.max_motion_time * 30
            if pred_joints.shape[0] < max_motion_len:
                # pad
                pad_length = max_motion_len - pred_joints.shape[0]
                pred_joints = torch.cat([pred_joints, torch.zeros((pad_length, pred_joints.shape[1], 3))], dim=0)
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
                if "avatarclip" in self.extra_path:
                    return_data["length"] = 60
        return return_data
