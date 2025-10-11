import torch
import torch.nn as nn
import importlib
import hmr4d.network.mdm.statistics  # example stats_module
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, compute_root_quaternion_ay
from hmr4d.model.supermotion.utils.motion3d_endecoder import EnDecoderBase, SMPLEnDecoder
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from einops import rearrange, einsum
from hmr4d.utils.smplx_utils import make_smplx
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.augment_noisy_pose import gaussian_augment
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


# transl_vel + global_orient + body_pose + betas + foot_contact
class SMPLRelVecV6EnDecoder(SMPLEnDecoder):
    def __init__(self, stats_module, stats_name, stats_incam_name, forward_func="fk", noise_pose_k=0):
        self.noise_pose_k = noise_pose_k
        super().__init__(stats_module, stats_name, forward_func)
        Log.info(f"We use noisy level k = {noise_pose_k} for local pose.")
        try:
            stats = getattr(importlib.import_module(stats_module), stats_incam_name)
            Log.info(f"We use {stats_incam_name} for pose in camera statistics!")
            self.register_buffer("poseincam_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("poseincam_std", torch.tensor(stats["std"]).float(), False)
        except Exception as e:
            print(e)
            Log.info(f"Cannot find {stats_incam_name} in {stats_module}! Use zero as mean, one as std!")
            self.register_buffer("poseincam_mean", torch.zeros(9).float(), False)
            self.register_buffer("poseincam_std", torch.ones(9).float(), False)

    def get_transl_vel(self, transl, global_orient):
        """
        transl velocity is in local coordinate after global_orient
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
        Returns:
            trans_vel: (B, L, 3)
        """
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = transl[:, 1:] - transl[:, :-1]  # (B, L-1, 3)
        trans_vel = torch.cat([trans_vel, trans_vel[:, -1:]], dim=1)  # (B, L, 3)
        trans_vel = trans_vel[:, :, None]  # (B, L, 1, 3)
        trans_vel = matrix.get_relative_direction_to(trans_vel, global_orient_mat)  # (B, L, 1, 3)
        trans_vel = trans_vel[:, :, 0]  # (B, L, 3)

        return trans_vel

    def encode(self, data):
        """
        Args:
            data (dict):
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
                skeleton: (B, J, 3)
                gender: (B), default=None, "neutral" or "male" or "female"
                modeltype: (B), default=None, "smplx" or "smplh"
                length: (B), default=None, effective length of each sample
                "global_orient_incam": (B, L, 3)
        Returns:
            x: (B, C=155, L)
        """
        transl = data["transl"]  # (B, L, 3)
        global_orient = data["global_orient"]  # (B, L, 3)
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = transl.shape
        if "betas" not in data.keys() or data["betas"] == None:
            betas = self.betas.clone()
        else:
            betas = data["betas"]
        betas = betas[:, None].expand(B, L, -1)  # (B, L, 10)
        betas = betas.to(transl.device)

        global_orient_rot6d, body_pose_rot6d = self.smplpose2smplrot6d(global_orient, body_pose)
        # (B, L, 6)             , (B, L, 126)
        joints = self.forward_func(transl, global_orient, body_pose, betas)[0]  # (B, L, J, 3)

        transl_vel = self.get_transl_vel(transl, global_orient)  # (B, L, 3)

        contact_label = self.detect_foot_contact(joints)  # (B, L, 4)

        x = torch.cat(
            [
                transl_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 9:135
                contact_label,  # (B, L, 4) -> 135:139
                betas,  # (B, L, 10) -> 139:149
            ],
            dim=-1,
        )  # (B, L, 149)

        x = self.norm(x)

        global_orient_incam_aa = data["global_orient_incam"]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)

        x_incam = torch.cat(
            [
                global_orient_incam_rot6d,  # (B, L, 6) -> 0:6 / 149:155
            ],
            dim=-1,
        )  # (B, L, 6)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = (x_incam - poseincam_mean) / poseincam_std  # (B, L, C)
        x_incam = x_incam.permute(0, 2, 1)  # (B, C, L)

        x = torch.cat([x, x_incam], dim=1)  # (B, 155, L)

        # length = data.get("length", None)
        # x = self.set_default_padding(x, length)
        return x

    def decode(self, x):
        """
        deocde with gt first frame transl
        Args:
            x: (B, C=155, L)
        Returns:
            output (dict): in ayfz coordinate
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
            data: (B, L, c=69),
        """
        denorm_x = self.denorm(x[:, :149])  # (B, L, C)
        B, L, _ = denorm_x.shape

        ayfz_global_orient_rot6d = denorm_x[..., 3:9]  # (B, L, 6)
        ayfz_global_orient_rotmat = rotation_6d_to_matrix(ayfz_global_orient_rot6d)  # (B, L, 3, 3)

        vel_global_orient_rotmat = torch.cat(
            [ayfz_global_orient_rotmat[:, :1], ayfz_global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        ayfz_transl_vel = denorm_x[..., 0:3]  # (B, L, 3)
        ayfz_transl = torch.zeros_like(ayfz_transl_vel)  # (B, L, 3)
        ayfz_transl[:, 1:] = ayfz_transl_vel[:, :-1]  # (B, L, 3)
        ayfz_transl = ayfz_transl[:, :, None]  # (B, L, 1, 3)
        ayfz_transl = matrix.get_position_from_rotmat(ayfz_transl, vel_global_orient_rotmat)  # (B, L, 1, 3)
        ayfz_transl = ayfz_transl[:, :, 0]  # (B, L, 3)
        ayfz_transl = torch.cumsum(ayfz_transl, dim=1)  # (B, L, 3)

        if hasattr(self, "ayfz_transl") and self.ayfz_transl is not None:
            gt_ayfz_transl = self.ayfz_transl
            ayfz_transl0 = gt_ayfz_transl[:, :1]  # (B, 1, 3)
            ayfz_transl = ayfz_transl + ayfz_transl0  # (B, L, 3)
        else:
            print("No gt_ayfz_transl, use zero as first frame transl.")

        body_pose_rot6d = denorm_x[..., 9:135]
        ayfz_global_orient, body_pose = self.smplrot6d2smplpose(ayfz_global_orient_rot6d, body_pose_rot6d)
        # (B, L, 3)       , (B, L, 63)

        contact_label = denorm_x[..., 135:139]  # (B, L, 4)

        betas = denorm_x[:, :, 139:149]  # (B, L, 10)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = x[:, 149:155, :]  # (B, 6, L)
        x_incam = x_incam.permute(0, 2, 1)  # (B, L, 6)
        x_incam = x_incam * poseincam_std + poseincam_mean
        global_orient_incam_rot6d = x_incam[..., :6]  # (B, L, 6)
        global_orient_incam_aa = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_incam_rot6d))  # (B, L, 3)

        output = {
            "transl": ayfz_transl,
            "global_orient": ayfz_global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "contact_label": contact_label,
            "global_orient_incam": global_orient_incam_aa,
        }

        return output

    def get_noisyobs(self, data, return_type="r6d"):
        """
        Noisy observation contains local pose with noise
        Args:
            data (dict):
                body_pose: (B, L, J*3) or (B, L, J, 3)
        Returns:
            noisy_bosy_pose: (B, L, J, 6) or (B, L, J, 3) or (B, L, 3, 3) depends on return_type
        """
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = body_pose.shape
        body_pose = body_pose.reshape(B, L, -1, 3)

        # (B, L, J, C)
        return_mapping = {"R": 0, "r6d": 1, "aa": 2}
        return_id = return_mapping[return_type]
        noisy_bosy_pose = gaussian_augment(body_pose, self.noise_pose_k, to_R=True)[return_id]
        return noisy_bosy_pose

    def normalize_local_pose_r6d(self, x, length=None):
        """
        normalize local pose r6d
        Args:
            x: (B, L, J*6)
        Returns:
            x: (B, L, J*6)
        """
        B, L = x.shape[:2]
        x = x.reshape(B, L, -1)

        noisyobs_mean, noisyobs_std = self.get_noisyobs_meanstd()

        x = (x - noisyobs_mean) / noisyobs_std  # (B, L, C)

        # x = x.permute(0, 2, 1)  # (B, C, L)
        # x = self.set_default_padding(x, length)
        # x = x.permute(0, 2, 1)  # (B, L, C)
        return x

    def get_noisyobs_meanstd(self):
        noisyobs_mean = self.mean[9:135]
        noisyobs_std = self.std[9:135]
        return noisyobs_mean, noisyobs_std

    def encode_a_decode(self, data, ayfz_transl_vel, ayfz_global_orient_rot6d):
        """Used in guidance
        Especially designed to approximately make: encode_a_decode(decode(x)) == x ,
        But this will never happen, since network prediction is not consistent within the representation
        """
        B, L = data["body_pose"].shape[:2]
        body_pose = data["body_pose"]
        betas = data["betas"]  # (B, L, 10)

        body_pose_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose.reshape(B, L, 21, 3)))
        body_pose_rot6d = body_pose_rot6d.reshape(B, L, -1)
        local_joints = self.fk_v2(body_pose, betas)
        local_joints = local_joints - local_joints[..., :1, :]  # remove root
        local_joints = local_joints[:, :, 1:, :].reshape(B, L, -1)

        x = torch.cat(
            [
                ayfz_transl_vel,  # (B, L, 3) -> 0:3
                ayfz_global_orient_rot6d,  # (B, L, 6) -> 3:9
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 9:135
                data["contact_label"],  # (B, L, 4) -> 135:139
                betas,  # (B, L, 10) -> 139:149
            ],
            dim=-1,
        )  # (B, L, 149)

        x = self.norm(x)  # (B, 149, L)

        # Incam params in x (Please fix this)
        global_orient_incam_aa = data["global_orient_incam"]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)

        x_incam = torch.cat(
            [
                global_orient_incam_rot6d,  # (B, L, 6) -> 0:6 / 149:155
            ],
            dim=-1,
        )  # (B, L, 6)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = (x_incam - poseincam_mean) / poseincam_std  # (B, L, C)
        x_incam = x_incam.permute(0, 2, 1)  # (B, C, L)

        x = torch.cat([x, x_incam], dim=1)  # (B, 155, L)

        return x
