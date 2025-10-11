import os
from pathlib import Path
from tqdm import trange
import numpy as np
import torch
from torch.utils import data
import codecs as cs
import random

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.pylogger import Log
from hmr4d.utils.hml3d.utils import standardize_motion
from hmr4d.utils.hml3d.utils import convert_motion_to_hmlvec263
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
import hmr4d.dataset.amass.amass_motion2d as amass_motion2d
from hmr4d.dataset.supermotion.collate import before_collate_motionsmpl
from hmr4d.dataset.hml3d.hml3d_utils import upsample_motion
from hmr4d.utils.camera_utils import get_camera_mat_zface, cartesian_to_spherical
import hmr4d.utils.matrix as matrix
from hmr4d.utils.geo_transform import project_p2d, cvt_to_bi01_p2d
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx


class BaseSMDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        target_fps=30,
        original_fps=30,
        original_coord="ay",
        min_motion_time=2,
        max_motion_time=10,
        sample_interval_time=1,
        limit_size=None,
    ):
        super().__init__()
        """Use SMPL pose"""
        self.root = Path(root)
        self.split = split
        self.target_fps = target_fps
        self.original_fps = original_fps
        self.original_coord = original_coord

        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.sample_interval_time = sample_interval_time
        self.limit_size = limit_size
        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        Log.info(f"Loading from {self.root / 'smplpose.pth'}...")
        self.motion_files = torch.load(self.root / "smplpose.pth")
        # Dict of {"gender": str,
        #          "pose": tensor(F, J, 3),
        #          "trans": tensor(F, 3),
        #          "beta": tensor(1, 10),
        #          "model": str, smplh/smplx
        #         }
        seqs = list(self.motion_files.keys())
        Log.info(f"Total number of sequences in {self.root}: {len(seqs)}")
        self.seqs = seqs

    def _get_idx2meta(self):
        idx2meta = []
        min_motion_len = self.original_fps * self.min_motion_time
        max_motion_len = self.original_fps * self.max_motion_time
        sample_interval = self.original_fps * self.sample_interval_time
        for k in self.seqs:
            m = self.motion_files[k]
            seq_length = len(m["trans"])
            max_sample_l = seq_length - max_motion_len + sample_interval
            max_sample_l = max(max_sample_l, 1)
            for start in range(0, max_sample_l, sample_interval):
                end = start + max_motion_len  # [start, end)
                if end > seq_length:
                    end = seq_length
                if end - start < min_motion_len and start != 0:
                    continue
                if end - start < 10:
                    # Filter too short sequences
                    continue
                idx2meta.append((k, start, end))
        self.idx2meta = idx2meta

        Log.info(
            f"Create {len(self.idx2meta)} sequences ({self.original_fps} FPS) with sample interval {sample_interval} for {self.root}"
        )

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start, end = meta
        # seq_name = "Minputs/amass/smplhg_raw/DanceDB/20150927_VasoAristeidou/Vasso_Bored_v1_01_poses.npz"
        # start = 0
        # end = -1
        motion = self.motion_files[seq_name]
        # Dict of {"gender": str,
        #          "pose": tensor(F, 63),
        #          "trans": tensor(F, 3),
        #          "beta": tensor(10),
        #          "skeleton": tensor(J, 3),
        #          "model": str, smplh/smplx
        #         }
        sampled_motion = {}

        for k, v in motion.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                sampled_motion[k] = v[start:end]
            else:
                sampled_motion[k] = v

        mlength = len(sampled_motion["pose"])
        min_motion_len = self.original_fps * self.min_motion_time
        max_motion_len = self.original_fps * self.max_motion_time

        min_length = min(min_motion_len, mlength - 1)
        length = np.random.randint(min_length, min(max_motion_len, mlength))  # random
        start = np.random.randint(0, mlength - length)
        end = start + length

        subsampled_motion = {}
        for k, v in sampled_motion.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                subsampled_motion[k] = v[start:end]
                if self.target_fps != self.original_fps:
                    # TODO: Simply linearly upsample is not good for axis-angle
                    subsampled_motion[k] = upsample_motion(subsampled_motion[k], self.original_fps, self.target_fps)
            else:
                subsampled_motion[k] = v

        return subsampled_motion

    def _process_data(self, data, idx):
        if self.original_coord == "ay":
            # ay to ay, do noting
            rot_mat = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 0.0]))  # 3, 3
        elif self.original_coord == "ayn":
            # ay negative to ay
            rot_mat = axis_angle_to_matrix(torch.tensor([0, 0, torch.pi]))  # 3, 3
        elif self.original_coord == "az":
            # az to ay
            rot_mat = axis_angle_to_matrix(torch.tensor([-torch.pi / 2, 0, 0]))  # 3, 3
        elif self.original_coord == "ax":
            # ax to ay
            rot_mat = axis_angle_to_matrix(torch.tensor([0, 0, -torch.pi / 2]))  # NOTE: never checked this
        global_orient = data["pose"][:, :3]
        global_orient_mat = axis_angle_to_matrix(global_orient)  # F, 3, 3
        global_orient_mat = matrix.get_mat_BfromA(rot_mat[None], global_orient_mat)
        ay_global_orient = matrix_to_axis_angle(global_orient_mat)
        trans = data["trans"]  # F, 3
        ay_trans = matrix.get_position_from_rotmat(trans, rot_mat)

        smplpose = torch.cat([ay_trans, ay_global_orient, data["pose"][:, 3:]], dim=-1)  # (F, 3+3+21*3=69)

        skeleton = data["skeleton"]  # J, 3
        beta = data["beta"]  # 10

        length = trans.shape[0]

        return_data = {
            "smplpose": smplpose,  # (F, 69)
            "length": length,
            "skeleton": skeleton,  # (J, 3)
            "beta": beta,  # (10)
            "gender": data["gender"],  # str: neutral, male, female
            "model": data["model"],  # str: smplh, smplx
        }
        return_data = before_collate_motionsmpl(
            return_data,
            max_len=self.target_fps * self.max_motion_time,
        )

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)


class FimgDictCombinedNpzs:
    def __init__(self, folder, npz_names):
        """
        Lazy loading by using npz. Support multiple npz files.
        """
        self.img_keys_to_npzid = {}
        self.npzFiles = []  # TODO: open npzFile here + DDP causes CRC-32 error
        self.npz_fpaths = []
        for i, npz in enumerate(npz_names):
            npz_file = np.load(folder / npz, allow_pickle=True)
            for k in npz_file.keys():
                self.img_keys_to_npzid[k] = i
            self.npz_fpaths.append(folder / npz)

    def keys(self):
        return self.img_keys_to_npzid.keys()

    def __getitem__(self, key):
        i = self.img_keys_to_npzid[key]
        return np.load(self.npz_fpaths[i], allow_pickle=True)[key].item()
