import torch
import torch.nn as nn
import importlib
import hmr4d.network.mdm.statistics  # example stats_module
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, compute_root_quaternion_ay
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


# fmt: off
class EnDecoderBase(nn.Module):
    def __init__(self, stats_module, stats_name) -> None:
        super().__init__()
        try:
            stats = getattr(importlib.import_module(stats_module), stats_name)
            Log.info(f"We use {stats_name} for statistics!")
            self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("std", torch.tensor(stats["std"]).float(), False)
        except Exception as e:
            print(e)
            Log.info(f"Cannot find {stats_name} in {stats_module}! Use zero as mean, one as std!")
            self.register_buffer("mean", torch.zeros(1).float(), False)
            self.register_buffer("std", torch.ones(1).float(), False)

    def encode(self, motion, length=None): raise NotImplemented  # (B, L, J, 3) -> (B, L, D)  TODO: Need discussion
    def decode(self, x): raise NotImplemented   # (B, L, D) -> (B, L, J, 3)
# fmt: on


# Absolute translation + absolute global_orient
class SMPLEnDecoder(EnDecoderBase):
    feet_ind = [8, 11, 7, 10]  # fid_r, fid_l = [8, 11], [7, 10]
    contact_thresh = 0.002

    def __init__(self, stats_module, stats_name, forward_func="fk"):
        """Input smplpose, output normalized smpl pose"""
        """smplpose includes: global translation(3), global orientation(3), body pose(63)"""
        super().__init__(stats_module, stats_name)
        self.mean = self.mean.reshape(-1)
        self.std = self.std.reshape(-1)
        if forward_func == "smpl":
            self.forward_func = self.smpl_forward
        elif forward_func == "fk":
            self.forward_func = self.fk_forward
        else:
            raise NotImplementedError
        self._build_smpl(forward_func)

    def decode_joints(self, x, *args, **kwargs):
        """
        decode joints from x
        Args:
            x: (B, C=135, L)
        Returns:
            joints: (B, L, J, 3), in ayfz coordinate
        """
        smplpose = self.decode(x, *args, **kwargs)
        joints = self.forward_func(smplpose)
        return joints

    def decode_joints_from_vel(self, x, *args, **kwargs):
        # smplpose representation does not have vel
        return self.decode_joints(x, *args, **kwargs)

    def decode_joints_from_root(self, x, *args, **kwargs):
        # smplpose representation does not have root
        return self.decode_joints(x, *args, **kwargs)

    def decode_joints_from_root_vel(self, x, *args, **kwargs):
        # smplpose representation does not have root vel
        return self.decode_joints(x, *args, **kwargs)

    def _build_smpl(self, forward_func):
        # SMPL forward is too slow
        smplh_model = {
            "male": make_smplx("rich-smplh", gender="male"),
            "female": make_smplx("rich-smplh", gender="female"),
            "neutral": make_smplx("rich-smplh", gender="neutral"),
        }
        smplx_model = {
            "male": make_smplx("rich-smplx", gender="male"),
            "female": make_smplx("rich-smplx", gender="female"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
        }
        self.body_model = {"smplh": smplh_model, "smplx": smplx_model}
        skeleton = smplh_model["neutral"]().joints[0, :22]  # 22, 3
        parents = smplh_model["neutral"].bm.parents[:22]  # 22 int
        local_skeleton = skeleton - skeleton[parents]
        local_skeleton = torch.cat([skeleton[:1], local_skeleton[1:]])  # 22, 3
        local_skeleton = local_skeleton[None]  # 1, 22, 3
        beta = torch.zeros(1, 10)  # 1, 10
        self.register_buffer("beta", beta, False)
        self.gender = ["neutral"]
        self.modeltype = ["smplh"]
        self.register_buffer("local_skeleton", local_skeleton, False)
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def set_skeleton(self, skeleton):
        """
        Args:
            skeleton: (B, J, 3)
        """
        local_skeleton = skeleton - skeleton[:, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :1], local_skeleton[:, 1:]], dim=1)
        self.register_buffer("local_skeleton", local_skeleton, False)

    def set_beta(self, beta):
        """
        Args:
            beta: (B, 10)
        """
        self.register_buffer("beta", beta, False)

    def set_gender(self, gender):
        """
        Args:
            gender: (B) str
        """
        self.gender = gender

    def set_modeltype(self, modeltype):
        """
        Args:
            modeltype: (B) str
        """
        self.modeltype = modeltype

    def smpl_forward(self, data):
        """
        Args:
            data: (B, L, 69)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = data.shape
        trans = data[:, :, :3]
        global_orient = data[:, :, 3:6]
        body_pose = data[:, :, 6:]
        beta = self.beta.clone()[:, None].expand(B, L, -1)
        beta = beta.to(data.device)
        all_joints = []
        for i in range(B):
            if i >= len(self.modeltype):
                modeltype = self.modeltype[-1]
                gender = self.gender[-1]
            else:
                modeltype = self.modeltype[i]
                gender = self.gender[i]
            smpl_model = self.body_model[modeltype][gender].to(data.device)

            beta_ = beta[i]
            trans_ = trans[i]
            global_orient_ = global_orient[i]
            body_pose_ = body_pose[i]
            smpl_out = smpl_model(
                betas=beta_.reshape(-1, 10),
                transl=trans_.reshape(-1, 3),
                global_orient=global_orient_.reshape(-1, 3),
                body_pose=body_pose_.reshape(-1, 63),
            )
            joints = smpl_out.joints[:, :22]  # (L, J, 3)
            all_joints.append(joints)
        all_joints = torch.stack(all_joints, dim=0)
        return all_joints

    def fk_forward(self, data):
        """
        Args:
            data: (B, L, 69)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = data.shape
        trans = data[:, :, :3]
        rot = data[:, :, 3:].reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(rot)  # (B, L, 22, 3, 3)
        local_skeleton = self.local_skeleton[:, None].expand(B, L, -1, -1).clone()  # (B, L, 22, 3)
        local_skeleton = local_skeleton.to(data.device)
        local_skeleton[..., 0, :] = local_skeleton[..., 0, :] + trans  # B, L, 22, 3
        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3

        return joints

    def get_ayfz_joints(self, data):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            ayfz_joints: (B, L, 22, 3)
        """
        B, L, _ = data.shape
        joints = self.forward_func(data)  # (B, L, 22, 3)
        joints_firstframe = joints[:, 0]  # (B, 22, 3)
        T_ay2ayfz = compute_T_ayfz2ay(joints_firstframe, inverse=True)  # (B, 4, 4)
        ayfz_joints = apply_T_on_points(joints.reshape(B, -1, 3), T_ay2ayfz).reshape(B, L, -1, 3)  # B, L, 22, 3
        # put on the floor
        ayfz_joints_floor = ayfz_joints.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
        ayfz_joints[..., 1] = ayfz_joints[..., 1] - ayfz_joints_floor[:, None, None]
        return ayfz_joints

    def get_ayfz_smplpose(self, data):
        """
        Args:
            data: (B, L, c=69)
        Returns:
            ayfz_smplpose: (B, L, c=69)
            ayfz_joints: (B, L, J, 3)
        """
        B, L, _ = data.shape
        joints = self.forward_func(data)  # (B, L, 22, 3)

        if False:  # Visualize
            bid = 0
            gt_params = {
                "betas": self.beta[[bid]].expand(L, -1),  # (L, 10)
                "transl": data[bid, :, :3],  # (B, L, 3)
                "global_orient": data[bid, :, 3:6],  # (B, L, 3)
                "body_pose": data[bid, :, 6:],  # (B, L, 63)
            }
            smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()
            smplx_out = smplx_model(**gt_params)
            smplx_out_joints = smplx_out.joints[:, :22]  # (L, 22, 3)
            forward_func_joints = joints[bid]  # (L, 22, 3)

            wis3d = make_wis3d(name="debug-endecoder-forward")
            add_motion_as_lines(smplx_out_joints, wis3d, name="smplx_out_joints")
            add_motion_as_lines(forward_func_joints, wis3d, name="forward_func_joints")

            vertices = smplx_out.vertices  # (L, V, 3)
            for i in range(L):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(vertices[i], smplx_model.bm.faces, name="smplx_out_mesh")

        joints_firstframe = joints[:, 0]  # (B, 22, 3)
        T_ay2ayfz = compute_T_ayfz2ay(joints_firstframe, inverse=True)  # (B, 4, 4)
        ayfz_joints = apply_T_on_points(joints.reshape(B, -1, 3), T_ay2ayfz).reshape(B, L, -1, 3)  # B, L, 22, 3
        ayfz_joints_floor = ayfz_joints.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
        ayfz_joints[..., 1] = ayfz_joints[..., 1] - ayfz_joints_floor[:, None, None]

        # ay to ayfz trans
        trans = data[..., :3]  # (B, L, 3)
        global_orient = data[..., 3:6]  # (B, L, 3)

        # smpl has root offset, should also be transformed
        root_offset = self.local_skeleton[:, :1]  # (B, 1, 3)
        root_offset = root_offset.to(data.device)
        root_offset = root_offset.expand(B, L, -1).clone()  # (B, L, 3)
        trans_with_offset = trans + root_offset  # (B, L, 3)
        ayfz_trans_with_offset = matrix.get_position_from(trans_with_offset, T_ay2ayfz)  # (B, L, 3)
        ayfz_trans = ayfz_trans_with_offset - root_offset  # (B, L, 3)

        # put on the floor
        ayfz_trans[..., 1] = ayfz_trans[..., 1] - ayfz_joints_floor[:, None]

        # ay to ayfz rot
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        ayfz_global_orient_mat = matrix.get_mat_BfromA(T_ay2ayfz[:, None, :3, :3], global_orient_mat)
        ayfz_global_orient = matrix_to_axis_angle(ayfz_global_orient_mat)
        ayfz_smplpose = torch.cat([ayfz_trans, ayfz_global_orient, data[..., 6:]], dim=-1)  # (B, L, 69)
        return ayfz_smplpose, ayfz_joints

    def smplpose2smplrot6d(self, smplpose):
        """
        Args:
            data: (B, L, c=69)
        Returns:
            x: (B, L, C=135)
        """
        B, L, _ = smplpose.shape
        trans = smplpose[..., :3]
        rot = smplpose[..., 3:].reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(rot)  # (B, L, 22, 3, 3)
        rot6d = matrix_to_rotation_6d(rotmat)  # (B, L, 22, 6)
        rot6d = rot6d.reshape(B, L, -1)  # (B, L, 132)
        smplrot6d = torch.cat([trans, rot6d], dim=-1)  # (B, L, 135)
        return smplrot6d

    def smplrot6d2smplpose(self, smplrot6d):
        """
        Args:
            data: (B, L, c=135)
        Returns:
            x: (B, L, C=69)
        """
        B, L, _ = smplrot6d.shape
        trans = smplrot6d[..., :3]
        rot6d = smplrot6d[..., 3:].reshape(B, L, -1, 6)
        rotmat = rotation_6d_to_matrix(rot6d)  # (B, L, 22, 3, 3)
        rot = matrix_to_axis_angle(rotmat)  # (B, L, 22, 3)
        rot = rot.reshape(B, L, -1)  # (B, L, 66)
        smplpose = torch.cat([trans, rot], dim=-1)  # (B, L, 69)
        return smplpose

    def set_default_padding(self, x, length=None):
        """
        Args:
            x: (B, C, L)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=135, L)
        """
        if length is not None:
            for i, l in enumerate(length):
                x[i, :, l:] = 0.0
        return x

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=135, L)
        """
        data = data.clone()
        ayfz_smplpose, _ = self.get_ayfz_smplpose(data)  # (B, L, c)
        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)
        x = (ayfz_smplrot6d - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def decode(self, x):
        """
        Args:
            x: (B, C=135, L)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """
        smplrot6d = self.denorm(x)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose

    def detect_foot_contact(self, ayfz_joints):
        """
        Args:
            ayfz_joints: (B, L, J, 3)
        Returns:
            x: (B, L, 4)
        """

        ayfz_feet = ayfz_joints[..., self.feet_ind, :]  # (B, L, 4, 3)
        ayfz_feet_vel = ayfz_feet[:, 1:] - ayfz_feet[:, :-1]  # (B, L - 1, 4, 3)
        ayfz_feet_l2dis = torch.norm(ayfz_feet_vel, dim=-1)  # (B, L - 1, 4)
        ayfz_feet_l2dis = torch.cat([ayfz_feet_l2dis, ayfz_feet_l2dis[:, -1:]], dim=-2)  # (B, L, 4)
        contact_label = ayfz_feet_l2dis < self.contact_thresh  # (B, L, 4)
        contact_label = contact_label.float()  # (B, L, 4)
        return contact_label

    def get_foot_vel(self, joints_vel):
        """
        Args:
            joints_vel: (B, L, J, 3) in global coordinate
        Returns:
            x: (B, L, 4, 3)
        """
        return joints_vel[..., self.feet_ind, :]

    def denorm(self, x):
        """
        Args:
            x: (B, C=135, L)
        Returns:
            x: (B, L, C=135)
        """
        x = x.permute(0, 2, 1)  # (B, L, 273)
        x = x * self.std + self.mean  # (B, L, 273)
        return x


class SMPLVecEnDecoder(SMPLEnDecoder):
    def get_local_joints(self, ayfz_joints):
        """
        joints velocity is in local ayfz coordinate
        Args:
            ayfz_joints: (B, L, J, 3)
        Returns:
            ayfz_local_joints: (B, L, J, 3)
            ayfz_local_joints_vel: (B, L, J, 3)
        """
        B, L, J, _ = ayfz_joints.shape

        T_ay2ayfz_eachframe = compute_T_ayfz2ay(ayfz_joints.reshape(B * L, -1, 3), inverse=True)  # (B*L, 4, 4)
        T_ay2ayfz_eachframe = T_ay2ayfz_eachframe.reshape(B, L, 4, 4)
        ayfz_local_joints = matrix.get_position_from(ayfz_joints, T_ay2ayfz_eachframe)  # (B, L, J, 3)
        ayfz_joints_vel = ayfz_joints[:, 1:] - ayfz_joints[:, :-1]  # (B, L-1, J, 3)
        ayfz_joints_vel = torch.cat([ayfz_joints_vel, ayfz_joints_vel[:, -1:]], dim=1)  # (B, L, J, 3)
        ayfz_local_joints_vel = matrix.get_position_from_rotmat(ayfz_joints_vel, T_ay2ayfz_eachframe[..., :3, :3])
        return ayfz_local_joints, ayfz_local_joints_vel

    def get_trans_vel(self, trans, global_orient):
        """
        trans velocity is in local coordinate after global_orient
        Args:
            trans: (B, L, 3)
            global_orient: (B, L, 3)
        Returns:
            trans_vel: (B, L, 3)
        """
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = trans[:, 1:] - trans[:, :-1]  # (B, L-1, 3)
        trans_vel = torch.cat([trans_vel, trans_vel[:, -1:]], dim=1)  # (B, L, 3)
        trans_vel = trans_vel[:, :, None]  # (B, L, 1, 3)
        trans_vel = matrix.get_relative_direction_to(trans_vel, global_orient_mat)  # (B, L, 1, 3)
        trans_vel = trans_vel[:, :, 0]  # (B, L, 3)

        return trans_vel

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=273, L)
        """
        data = data.clone()
        ayfz_smplpose, ayfz_joints = self.get_ayfz_smplpose(data)  # (B, L, c), (B, L, J, 3)
        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)
        contact_label = self.detect_foot_contact(ayfz_joints)  # (B, L, 4)
        ayfz_local_joints, ayfz_local_joints_vel = self.get_local_joints(ayfz_joints)
        root_y = ayfz_local_joints[..., 0, [1]]  # (B, L, 1)
        root_xz_vel = ayfz_local_joints_vel[..., 0, [0, 2]]  # (B, L, 2)
        quat_root_y = compute_root_quaternion_ay(ayfz_joints)  # continuous quaternions while pytroch3d func does not
        ang_root_y = torch.asin(quat_root_y[..., 2:3])  # (B, L, 1)
        ang_root_y_vel = ang_root_y[:, 1:] - ang_root_y[:, :-1]  # (B, L - 1, 1)
        ang_root_y_vel = torch.cat([ang_root_y_vel, ang_root_y_vel[:, -1:]], dim=1)  # (B, L, 1)
        trans = ayfz_smplrot6d[..., :3]  # (B, L, 3)
        global_orient = ayfz_smplpose[..., 3:6]  # (B, L, 3)
        trans_vel = self.get_trans_vel(trans, global_orient)
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        global_orient_after_y = matrix.get_mat_BtoA(rotmat_root_y, global_orient_mat)  # (B, L, 3, 3)
        global_orient_after_y_rot6d = matrix_to_rotation_6d(global_orient_after_y)  # (B, L, 6)
        x = torch.cat(
            [
                root_xz_vel,  # (B, L, 2) -> 0:2
                ang_root_y,  # (B, L, 1) -> 2:3
                ang_root_y_vel,  # (B, L, 1) -> 3:4
                root_y,  # (B, L, 1) -> 4:5
                trans,  # (B, L, 3) -> 5:8
                trans_vel,  # (B, L, 3) -> 8:11
                global_orient_after_y_rot6d,  # (B, L, 6) -> 11:17
                ayfz_local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 17:80
                ayfz_local_joints_vel[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 80:143
                ayfz_smplrot6d[..., 9:],  # (B, L, (J-1)*6) -> 143:269
                contact_label,  # (B, L, 4) -> 269:273
            ],
            dim=-1,
        )  # (B, L, 273)

        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def get_joints_from_root(self, x):
        """
        get joints from root_y, root_xz_vel, root_ang_vel, local joints
        Args:
            x: (B, L, C=273)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """

        B, L, _ = x.shape
        root_xz_vel = x[..., 0:2]  # (B, L, 2)
        ang_root_y_vel = x[..., 3:4]  # (B, L, 1)
        root_y = x[..., 4:5]  # (B, L, 1)
        ayfz_local_joints = x[..., 17:80]  # (B, L, 21*3)
        ayfz_local_joints = ayfz_local_joints.reshape(B, L, -1, 3)

        ang_root_y = torch.zeros_like(x[..., 3:4])  # (B, L, 1)
        ang_root_y[:, 1:] = ang_root_y_vel[:, :-1]  # (B, L, 1)
        ang_root_y = torch.cumsum(ang_root_y, dim=-2)  # (B, L, 1)

        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)

        root_pos = torch.zeros(x.shape[:-1] + (3,), device=x.device)  # (B, L, 3)
        root_pos[..., 1:, [0, 2]] = root_xz_vel[:, :-1]  # (B, L, 3)
        rotmat_root = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        # velocity rotmat is in previous frame
        vel_rotmat_root = torch.cat([rotmat_root[:, :1], rotmat_root[:, :-1]], dim=1)  # (B, L, 3, 3)
        root_pos = matrix.get_position_from_rotmat(root_pos[..., None, :], vel_rotmat_root)[..., 0, :]  # (B, L, 3)
        root_pos = torch.cumsum(root_pos, dim=-2)  # (B, L, 3)
        root_pos[..., 1:2] = root_y  # (B, L, 3)

        ayfz_joints = matrix.get_position_from_rotmat(ayfz_local_joints, rotmat_root)  # (B, L, J-1, 3)
        ayfz_joints[..., 0] = ayfz_joints[..., 0] + root_pos[..., [0]]
        ayfz_joints[..., 2] = ayfz_joints[..., 2] + root_pos[..., [2]]
        ayfz_joints = torch.cat([root_pos[..., None, :], ayfz_joints], dim=-2)  # (B, L, J, 3)

        return ayfz_joints

    def get_joints_from_root_vel(self, x):
        """
        get joints from root_y, root_xz_vel, root_ang_vel, local joints(first frame), local_joints vel
        Args:
            x: (B, L, C=273)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """

        B, L, _ = x.shape
        root_xz_vel = x[..., 0:2]  # (B, L, 2)
        ang_root_y_vel = x[..., 3:4]  # (B, L, 1)
        root_y = x[..., 4:5]  # (B, L, 1)
        ayfz_local_joints = x[..., 17:80]  # (B, L, 21*3)
        ayfz_local_joints = ayfz_local_joints.reshape(B, L, -1, 3)
        ayfz_local_joints_vel = x[..., 80:143]  # (B, L, 21*3)
        ayfz_local_joints_vel = ayfz_local_joints_vel.reshape(B, L, -1, 3)

        ang_root_y = torch.zeros_like(x[..., 3:4])  # (B, L, 1)
        ang_root_y[:, 1:] = ang_root_y_vel[:, :-1]  # (B, L, 1)
        ang_root_y = torch.cumsum(ang_root_y, dim=-2)  # (B, L, 1)

        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)

        root_pos = torch.zeros(x.shape[:-1] + (3,), device=x.device)  # (B, L, 3)
        root_pos[..., 1:, [0, 2]] = root_xz_vel[:, :-1]  # (B, L, 3)
        rotmat_root = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        # velocity rotmat is in previous frame
        vel_rotmat_root = torch.cat([rotmat_root[:, :1], rotmat_root[:, :-1]], dim=1)  # (B, L, 3, 3)
        root_pos = matrix.get_position_from_rotmat(root_pos[..., None, :], vel_rotmat_root)[..., 0, :]  # (B, L, 3)
        root_pos = torch.cumsum(root_pos, dim=-2)  # (B, L, 3)
        root_pos[..., 1:2] = root_y  # (B, L, 3)

        ayfz_joints_firstframe = matrix.get_position_from_rotmat(
            ayfz_local_joints[:, 0], rotmat_root[:, 0]
        )  # (B, J-1, 3)
        ayfz_joints_firstframe[..., 0] = ayfz_joints_firstframe[..., 0] + root_pos[:, 0, [0]]
        ayfz_joints_firstframe[..., 2] = ayfz_joints_firstframe[..., 2] + root_pos[:, 0, [2]]
        ayfz_joints = torch.zeros_like(ayfz_local_joints)  # (B, L, J-1, 3)
        ayfz_joints[:, 1:] = ayfz_joints[:, 1:] + ayfz_local_joints_vel[:, :-1]
        ayfz_joints = matrix.get_position_from_rotmat(ayfz_joints, vel_rotmat_root)  # (B, L, J-1, 3)
        ayfz_joints[:, 0] = ayfz_joints[:, 0] + ayfz_joints_firstframe  # (B, L, J - 1, 3)
        ayfz_joints = torch.cumsum(ayfz_joints, dim=1)  # (B, L, J-1, 3)
        ayfz_joints = torch.cat([root_pos[..., None, :], ayfz_joints], dim=-2)  # (B, L, J, 3)

        return ayfz_joints

    def get_smplpose_from_vel(self, x):
        """
        get smplpose from first frame trans, trans vel, local pose
        Args:
            x: (B, L, C=273)
        Returns:
            smplpose: (B, L, 69) ayfz smpl pose
        """

        ang_root_y = x[..., 2:3]  # (B, L, 1)
        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        global_orient_after_y_rot6d = x[..., 11:17]  # (B, L, 6)
        global_orient_after_y_rotmat = rotation_6d_to_matrix(global_orient_after_y_rot6d)  # (B, L, 3, 3)
        global_orient_rotmat = matrix.get_mat_BfromA(rotmat_root_y, global_orient_after_y_rotmat)  # (B, L, 3, 3)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)
        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        abs_trans = x[..., 5:8]  # (B, L, 3)
        trans_vel = x[..., 8:11]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + abs_trans[:, :1]  # (B, L, 3)
        smpl_rot6d = x[..., 143:269]

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose

    def get_joints_from_vel(self, x):
        """
        get joints from first frame trans, trans vel, local pose
        Args:
            x: (B, L, C=273)
        Returns:
            joints: (B, L, J, 3) ayfz coordinate
        """
        smplpose = self.get_smplpose_from_vel(x)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)
        return joints

    def decode_joints_from_root(self, x):
        """
        decode joints from root_y, root_xz_vel, root_ang_vel, local joints
        Args:
            x: (B, C=273, L)
        Returns:
            ayfz_joints: (B, L, J, 3), in ayfz coordinate
        """
        x = self.denorm(x)

        ayfz_joints = self.get_joints_from_root(x)
        return ayfz_joints

    def decode_joints_from_root_vel(self, x):
        """
        decode joints from root_y, root_xz_vel, root_ang_vel, local joints(first frame), local joints vel
        Args:
            x: (B, C=273, L)
        Returns:
            ayfz_joints: (B, L, J, 3), in ayfz coordinate
        """
        x = self.denorm(x)

        ayfz_joints = self.get_joints_from_root_vel(x)
        return ayfz_joints

    def decode_joints_from_vel(self, x):
        """
        decode joints from first frame trans, trans vel, local pose
        Args:
            x: (B, C=273, L)
        Returns:
            ayfz_joints: (B, L, J, 3), in ayfz coordinate
        """
        x = self.denorm(x)

        ayfz_joints = self.get_joints_from_vel(x)
        return ayfz_joints

    def decode(self, x):
        """
        Args:
            x: (B, C=273, L)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """
        x = self.denorm(x)  # B, L, C

        ang_root_y = x[..., 2:3]  # (B, L, 1)
        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        global_orient_after_y_rot6d = x[..., 11:17]  # (B, L, 6)
        global_orient_after_y_rotmat = rotation_6d_to_matrix(global_orient_after_y_rot6d)  # (B, L, 3, 3)
        global_orient_rotmat = matrix.get_mat_BfromA(rotmat_root_y, global_orient_after_y_rotmat)  # (B, L, 3, 3)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)
        trans = x[..., 5:8]  # (B, L, 3)
        smpl_rot6d = x[..., 143:269]

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose


# Translation velocity (in global orient coordinate) + global_orient velocity
class SMPLRelVecEnDecoder(SMPLEnDecoder):
    def get_trans_vel(self, trans, global_orient_mat):
        """
        trans velocity is in local coordinate after global_orient
        Args:
            trans: (B, L, 3)
            global_orient: (B, L, 3, 3)
        Returns:
            trans_vel: (B, L, 3)
        """
        trans_vel = trans[:, 1:] - trans[:, :-1]  # (B, L-1, 3)
        trans_vel = torch.cat([trans_vel, trans_vel[:, -1:]], dim=1)  # (B, L, 3)
        trans_vel = trans_vel[:, :, None]  # (B, L, 1, 3)
        trans_vel = matrix.get_relative_direction_to(trans_vel, global_orient_mat)  # (B, L, 1, 3)
        trans_vel = trans_vel[:, :, 0]  # (B, L, 3)

        return trans_vel

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=202, L)
        """
        data = data.clone()
        ayfz_smplpose, ayfz_joints = self.get_ayfz_smplpose(data)  # (B, L, c), (B, L, J, 3)
        trans = ayfz_smplpose[..., :3]  # (B, L, 3)
        global_orient = ayfz_smplpose[..., 3:6]  # (B, L, 3)
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = self.get_trans_vel(trans, global_orient_mat)  # (B, L, 3)
        local_joints = ayfz_joints - ayfz_joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)
        global_orient_mat_relative = matrix.get_mat_BtoA(
            global_orient_mat[:, :-1], global_orient_mat[:, 1:]
        )  # (B, L-1, 3, 3)
        global_orient_rot6d_relative = matrix_to_rotation_6d(global_orient_mat_relative)  # (B, L-1, 6)
        global_orient_rot6d_relative = torch.cat(
            [global_orient_rot6d_relative, global_orient_rot6d_relative[:, -1:]], dim=1
        )  # (B, L, 6)

        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)

        contact_label = self.detect_foot_contact(ayfz_joints)  # (B, L, 4)

        x = torch.cat(
            [
                trans_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d_relative,  # (B, L, 6) -> 3:9
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 9:72
                ayfz_smplrot6d[..., 9:],  # (B, L, (J-1)*6) -> 72:198
                contact_label,  # (B, L, 4) -> 198:202
            ],
            dim=-1,
        )  # (B, L, 202)

        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def get_smplpose_from_vel(self, x, firstframe_smplpose=None):
        """
        get smplpose from first frame trans, trans vel, local pose
        Args:
            x: (B, L, C=202)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            smplpose: (B, L, 69) ayfz smpl pose
        """
        if firstframe_smplpose is None:
            B = x.shape[0]
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)
        global_orient0 = firstframe_smplpose[..., 3:6]  # (B, 1, 3)
        global_orient0_rotmat = axis_angle_to_matrix(global_orient0)  # (B, 1, 3, 3)

        global_orient_rot6d_relative = x[..., 3:9]  # (B, L, 6)
        global_orient_rotmat_relative = rotation_6d_to_matrix(global_orient_rot6d_relative)  # (B, L, 3, 3)
        global_orient_rotmat = torch.zeros_like(global_orient_rotmat_relative)
        global_orient_rotmat[:, :1] = global_orient0_rotmat
        for i in range(1, global_orient_rotmat_relative.shape[1]):
            global_orient_rotmat[:, i] = matrix.get_mat_BfromA(
                global_orient_rotmat[:, i - 1], global_orient_rotmat_relative[:, i - 1]
            )

        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)
        smpl_rot6d = x[..., 72:198]

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose

    def decode(self, x, firstframe_smplpose=None):
        """
        Args:
            x: (B, C=202, L)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """
        x = self.denorm(x)  # B, L, C

        smplpose = self.get_smplpose_from_vel(x, firstframe_smplpose)
        return smplpose

    def get_joints_from_root(self, x, firstframe_smplpose=None):
        """
        get joints from first frame trans, trans vel, local joints
        Args:
            x: (B, L, C=202)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            local_joints: (B, L, J, 3) ayfz joints
        """
        B, L, _ = x.shape
        if firstframe_smplpose is None:
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)
        global_orient0 = firstframe_smplpose[..., 3:6]  # (B, 1, 3)
        global_orient0_rotmat = axis_angle_to_matrix(global_orient0)  # (B, 1, 3, 3)

        global_orient_rot6d_relative = x[..., 3:9]  # (B, L, 6)
        global_orient_rotmat_relative = rotation_6d_to_matrix(global_orient_rot6d_relative)  # (B, L, 3, 3)
        global_orient_rotmat = torch.zeros_like(global_orient_rotmat_relative)
        global_orient_rotmat[:, :1] = global_orient0_rotmat
        for i in range(1, global_orient_rotmat_relative.shape[1]):
            global_orient_rotmat[:, i] = matrix.get_mat_BfromA(
                global_orient_rotmat[:, i - 1], global_orient_rotmat_relative[:, i - 1]
            )

        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)

        smpl_rot6d = torch.zeros_like(x[..., 72:198])
        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)

        root_pos = joints[:, :, :1]  # (B, L, 1, 3)

        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)

        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root_pos  # (B, L, J-1, 3)
        joints = torch.cat([root_pos, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def decode_joints_from_root(self, x, *args, **kwargs):
        """
        decode joints from first frame trans, trans vel, local joints
        Args:
            x: (B, C=202, L)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            local_joints: (B, L, J, 3) ayfz joints
        """
        x = self.denorm(x)
        return self.get_joints_from_root(x, *args, **kwargs)

    def get_fk_local_joints(self, smplpose):
        """
        get forward kinematics local joints from smpl pose
        Args:
            smplpose: (B, L, C=69)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """
        smplpose_local = smplpose.clone()
        smplpose_local[..., :6] = 0
        local_joints = self.forward_func(smplpose_local)  # (B, L, J, 3)
        # SMPL FK has a root offset
        local_joints = local_joints - local_joints[:, :, :1]  # (B, L, J, 3)
        return local_joints

    def get_fk_local_joints_from_x(self, x):
        """
        get forward kinematics local joints from local pose
        Args:
            x: (B, L, C=202)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        smpl_rot6d = x[..., 72:198]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        local_joints = self.forward_func(smplpose)  # (B, L, J, 3)
        # SMPL FK has a root offset
        local_joints = local_joints - local_joints[:, :, :1]  # (B, L, J, 3)
        return local_joints

    def decode_fk_local_joints(self, x):
        """
        decode forward kinematics local joints from local pose
        Args:
            x: (B, C=202, L)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        x = self.denorm(x)
        local_joints = self.get_fk_local_joints_from_x(x)

        return local_joints

    def get_fk_joints_from_x(self, x, gt_smplpose=None):
        """
        get forward kinematics joints from local pose and gt_transl, global_orient
        Args:
            x: (B, L, C=202)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """
        print("WARNING: DEPRECATED FUNCTION")
        B, L, _ = x.shape
        smpl_rot6d = x[..., 72:198]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)
        if gt_smplpose is not None:
            trans = gt_smplpose[..., :3]
            global_orient_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(gt_smplpose[..., 3:6]))

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)
        return joints

    def decode_fk_joints(self, x, gt_smplpose=None):
        """
        decode forward kinematics global joints from local pose and gt global
        Args:
            x: (B, C=202, L)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """

        x = self.denorm(x)
        joints = self.get_fk_joints_from_x(x, gt_smplpose)

        return joints

    def get_fk_local_joints_from_local(self, x):
        """
        get forward kinematics local joints from local joints
        Args:
            x: (B, L, C=202)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        local_joints = torch.cat([torch.zeros_like(local_joints[..., :1, :]), local_joints], dim=-2)  # (B, L, J, 3)
        return local_joints

    def decode_fk_local_joints_from_local(self, x):
        """
        decode forward kinematics local joints from local pose
        Args:
            x: (B, C=202, L)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        x = self.denorm(x)
        local_joints = self.get_fk_local_joints_from_local(x)

        return local_joints

    def get_fk_joints_from_local(self, x, gt_smplpose):
        """
        get forward kinematics global joints from local joints and gt transl, global_orient
        Args:
            x: (B, L, C=202)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """
        print("WARNING: DEPRECATED FUNCTION")
        B, L, _ = x.shape
        gt_joints = self.forward_func(gt_smplpose)  # (B, L, J, 3)
        root = gt_joints[:, :, :1]  # (B, L, 1, 3)
        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        global_orient = gt_smplpose[..., 3:6]
        global_orient_rotmat = axis_angle_to_matrix(global_orient)  # (B, L, 3)
        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root
        joints = torch.cat([root, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def decode_fk_joints_from_local(self, x, gt_smplpose):
        """
        decode forward kinematics global joints from local joints and gt transl, global_orient
        Args:
            x: (B, C=202, L)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)  in ayfz
        """
        print("WARNING: DEPRECATED FUNCTION")
        x = self.denorm(x)
        joints = self.get_fk_joints_from_local(x, gt_smplpose)

        return joints


# Translation velocity (in global orient coordinate) + y-axis rot vel + absolute global orient after y-axis rot
class SMPLRelVecV2EnDecoder(SMPLRelVecEnDecoder):
    def __init__(self, stats_module, stats_name, forward_func="fk", noise_pose_k=0):
        self.noise_pose_k = noise_pose_k
        super().__init__(stats_module, stats_name, forward_func)
        Log.info(f"We use noisy level k = {noise_pose_k} for local pose.")

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=203, L)
        """
        data = data.clone()
        ayfz_smplpose, ayfz_joints = self.get_ayfz_smplpose(data)  # (B, L, c), (B, L, J, 3)
        trans = ayfz_smplpose[..., :3]  # (B, L, 3)
        global_orient = ayfz_smplpose[..., 3:6]  # (B, L, 3)
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = self.get_trans_vel(trans, global_orient_mat)  # (B, L, 3)
        local_joints = ayfz_joints - ayfz_joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)

        quat_root_y = compute_root_quaternion_ay(ayfz_joints)  # continuous quaternions while pytroch3d func does not
        ang_root_y = torch.asin(quat_root_y[..., 2:3])  # (B, L, 1)
        ang_root_y_vel = ang_root_y[:, 1:] - ang_root_y[:, :-1]  # (B, L - 1, 1)
        ang_root_y_vel = torch.cat([ang_root_y_vel, ang_root_y_vel[:, -1:]], dim=1)  # (B, L, 1)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)
        global_orient_after_y = matrix.get_mat_BtoA(rotmat_root_y, global_orient_mat)  # (B, L, 3, 3)
        global_orient_after_y_rot6d = matrix_to_rotation_6d(global_orient_after_y)  # (B, L, 6)

        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)

        contact_label = self.detect_foot_contact(ayfz_joints)  # (B, L, 4)

        x = torch.cat(
            [
                trans_vel,  # (B, L, 3) -> 0:3
                ang_root_y_vel,  # (B, L, 1) -> 3:4
                global_orient_after_y_rot6d,  # (B, L, 6) -> 4:10
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 10:73
                ayfz_smplrot6d[..., 9:],  # (B, L, (J-1)*6) -> 73:199
                contact_label,  # (B, L, 4) -> 199:203
            ],
            dim=-1,
        )  # (B, L, 203)

        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def get_smplpose_from_vel(self, x, firstframe_smplpose=None):
        """
        get smplpose from first frame trans, trans vel, local pose
        Args:
            x: (B, L, C=203)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            smplpose: (B, L, 69) ayfz smpl pose
        """
        if firstframe_smplpose is None:
            B = x.shape[0]
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)

        ang_root_y = torch.zeros_like(x[..., 3:4])  # (B, L, 1)
        ang_root_y_vel = x[..., 3:4]  # (B, L, 1)
        ang_root_y[:, 1:] = ang_root_y_vel[:, :-1]  # (B, L, 1)
        ang_root_y = torch.cumsum(ang_root_y, dim=-2)  # (B, L, 1)

        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)

        global_orient_after_y_rot6d = x[..., 4:10]  # (B, L, 6)
        global_orient_after_y_rotmat = rotation_6d_to_matrix(global_orient_after_y_rot6d)  # (B, L, 3, 3)
        global_orient_rotmat = matrix.get_mat_BfromA(rotmat_root_y, global_orient_after_y_rotmat)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)
        smpl_rot6d = x[..., 73:199]

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose

    def decode(self, x, firstframe_smplpose=None):
        """
        Args:
            x: (B, C=203, L)
        Returns:
            data: (B, L, c=69), in ayfz coordinate
        """
        x = self.denorm(x)  # B, L, C

        smplpose = self.get_smplpose_from_vel(x, firstframe_smplpose)
        return smplpose

    def get_joints_from_root(self, x, firstframe_smplpose=None):
        """
        get joints from first frame trans, trans vel, local joints
        Args:
            x: (B, L, C=203)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            local_joints: (B, L, J, 3) ayfz joints
        """
        B, L, _ = x.shape
        if firstframe_smplpose is None:
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)

        ang_root_y = torch.zeros_like(x[..., 3:4])  # (B, L, 1)
        ang_root_y_vel = x[..., 3:4]  # (B, L, 1)
        ang_root_y[:, 1:] = ang_root_y_vel[:, :-1]  # (B, L, 1)
        ang_root_y = torch.cumsum(ang_root_y, dim=-2)  # (B, L, 1)

        quat_root_y = torch.zeros(x.shape[:-1] + (4,), device=x.device)  # (B, L, 4)
        quat_root_y[..., :1] = torch.cos(ang_root_y)  # (B, L, 4)
        quat_root_y[..., 2:3] = torch.sin(ang_root_y)  # (B, L, 4)
        rotmat_root_y = quaternion_to_matrix(quat_root_y)  # (B, L, 3, 3)

        global_orient_after_y_rot6d = x[..., 4:10]  # (B, L, 6)
        global_orient_after_y_rotmat = rotation_6d_to_matrix(global_orient_after_y_rot6d)  # (B, L, 3, 3)
        global_orient_rotmat = matrix.get_mat_BfromA(rotmat_root_y, global_orient_after_y_rotmat)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_rotmat)  # (B, L, 6)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)

        smpl_rot6d = torch.zeros_like(x[..., 73:199])
        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)

        root_pos = joints[:, :, :1]  # (B, L, 1, 3)

        local_joints = x[..., 10:73]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)

        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root_pos  # (B, L, J-1, 3)
        joints = torch.cat([root_pos, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def decode_joints_from_root(self, x, *args, **kwargs):
        """
        decode joints from first frame trans, trans vel, local joints
        Args:
            x: (B, C=202, L)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            local_joints: (B, L, J, 3) ayfz joints
        """
        x = self.denorm(x)
        return self.get_joints_from_root(x, *args, **kwargs)

    def get_fk_local_joints(self, smplpose):
        """
        get forward kinematics local joints from smpl pose
        Args:
            smplpose: (B, L, C=69)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """
        smplpose_local = smplpose.clone()
        smplpose_local[..., :6] = 0
        local_joints = self.forward_func(smplpose_local)  # (B, L, J, 3)
        # SMPL FK has a root offset
        local_joints = local_joints - local_joints[:, :, :1]  # (B, L, J, 3)
        return local_joints

    def get_fk_local_joints_from_x(self, x):
        """
        get forward kinematics local joints from local pose
        Args:
            x: (B, L, C=203)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        smpl_rot6d = x[..., 73:199]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        local_joints = self.forward_func(smplpose)  # (B, L, J, 3)
        # SMPL FK has a root offset
        local_joints = local_joints - local_joints[:, :, :1]  # (B, L, J, 3)
        return local_joints

    def decode_fk_local_joints(self, x):
        """
        decode forward kinematics local joints from local pose
        Args:
            x: (B, C=203, L)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        x = self.denorm(x)
        local_joints = self.get_fk_local_joints_from_x(x)

        return local_joints

    def get_fk_joints_from_x(self, x, gt_smplpose=None):
        """
        get forward kinematics joints from local pose and gt_transl, global_orient
        Args:
            x: (B, L, C=203)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """
        print("WARNING: DEPRECATED FUNCTION")
        B, L, _ = x.shape
        smpl_rot6d = x[..., 73:199]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)
        if gt_smplpose is not None:
            trans = gt_smplpose[..., :3]
            global_orient_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(gt_smplpose[..., 3:6]))

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)
        return joints

    def decode_fk_joints(self, x, gt_smplpose=None):
        """
        decode forward kinematics global joints from local pose and gt global
        Args:
            x: (B, C=203, L)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """

        x = self.denorm(x)
        joints = self.get_fk_joints_from_x(x, gt_smplpose)

        return joints

    def get_fk_local_joints_from_local(self, x):
        """
        get forward kinematics local joints from local joints
        Args:
            x: (B, L, C=203)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        local_joints = x[..., 10:73]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        local_joints = torch.cat([torch.zeros_like(local_joints[..., :1, :]), local_joints], dim=-2)  # (B, L, J, 3)
        return local_joints

    def decode_fk_local_joints_from_local(self, x):
        """
        decode forward kinematics local joints from local pose
        Args:
            x: (B, C=203, L)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        x = self.denorm(x)
        local_joints = self.get_fk_local_joints_from_local(x)

        return local_joints

    def get_fk_joints_from_local(self, x, gt_smplpose):
        """
        get forward kinematics global joints from local joints and gt transl, global_orient
        Args:
            x: (B, L, C=203)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """

        B, L, _ = x.shape
        gt_joints = self.forward_func(gt_smplpose)  # (B, L, J, 3)
        root = gt_joints[:, :, :1]  # (B, L, 1, 3)
        local_joints = x[..., 10:73]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        global_orient = gt_smplpose[..., 3:6]
        global_orient_rotmat = axis_angle_to_matrix(global_orient)  # (B, L, 3)
        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root
        joints = torch.cat([root, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def decode_fk_joints_from_local(self, x, gt_smplpose):
        """
        decode forward kinematics global joints from local joints and gt transl, global_orient
        Args:
            x: (B, C=203, L)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)  in ayfz
        """

        x = self.denorm(x)
        joints = self.get_fk_joints_from_local(x, gt_smplpose)

        return joints

    def get_noisy_meanstd(self):

        noisyobs_mean = self.mean[73:199]
        noisyobs_std = self.std[73:199]
        return noisyobs_mean, noisyobs_std

    def get_noisyobs(self, data, length=None):
        """
        Noisy observation contains local pose with noise
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, L, C=126)
        """
        data = data.clone()
        B, L, _ = data.shape
        if data.shape[-1] == 69:
            localpose_aa = data[..., 6:].reshape(B, L, -1, 3)
        elif data.shape[-1] == 63:
            localpose_aa = data.reshape(B, L, -1, 3)

        new_localpose_r6d = gaussian_augment(localpose_aa, self.noise_pose_k, to_R=True)[1]  # (B, L, J, 6)

        return self.normalize_local_pose_r6d(new_localpose_r6d, length)

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

        noisyobs_mean, noisyobs_std = self.get_noisy_meanstd()

        x = (x - noisyobs_mean) / noisyobs_std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        x = x.permute(0, 2, 1)  # (B, L, C)
        return x


# Translation velocity (in global orient coordinate) + absolute global orient
class SMPLRelVecV3EnDecoder(SMPLRelVecV2EnDecoder):

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=202, L)
        """
        data = data.clone()
        ayfz_smplpose, ayfz_joints = self.get_ayfz_smplpose(data)  # (B, L, c), (B, L, J, 3)
        trans = ayfz_smplpose[..., :3]  # (B, L, 3)
        global_orient = ayfz_smplpose[..., 3:6]  # (B, L, 3)
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = self.get_trans_vel(trans, global_orient_mat)  # (B, L, 3)
        local_joints = ayfz_joints - ayfz_joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)

        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_mat)  # (B, L, 6)

        contact_label = self.detect_foot_contact(ayfz_joints)  # (B, L, 4)

        x = torch.cat(
            [
                trans_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 9:72
                ayfz_smplrot6d[..., 9:],  # (B, L, (J-1)*6) -> 72:198
                contact_label,  # (B, L, 4) -> 198:202
            ],
            dim=-1,
        )  # (B, L, 202)

        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def get_smplpose_from_vel(self, x, firstframe_smplpose=None):
        """
        get smplpose from first frame trans, trans vel, local pose
        Args:
            x: (B, L, C=202)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            smplpose: (B, L, 69) ayfz smpl pose
        """
        if firstframe_smplpose is None:
            B = x.shape[0]
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)

        global_orient_rot6d = x[..., 3:9]  # (B, L, 6)
        global_orient_rotmat = rotation_6d_to_matrix(global_orient_rot6d)  # (B, L, 3, 3)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)
        smpl_rot6d = x[..., 72:198]

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        return smplpose

    def get_joints_from_root(self, x, firstframe_smplpose=None):
        """
        get joints from first frame trans, trans vel, local joints
        Args:
            x: (B, L, C=202)
            firstframe_smplpose: (B, 1, 69)
        Returns:
            local_joints: (B, L, J, 3) ayfz joints
        """
        B, L, _ = x.shape
        if firstframe_smplpose is None:
            firstframe_smplpose = torch.zeros((B, 1, 69), device=x.device)

        trans0 = firstframe_smplpose[..., :3]  # (B, 1, 3)

        global_orient_rot6d = x[..., 3:9]  # (B, L, 6)
        global_orient_rotmat = rotation_6d_to_matrix(global_orient_rot6d)  # (B, L, 3, 3)

        vel_global_orient_rotmat = torch.cat(
            [global_orient_rotmat[:, :1], global_orient_rotmat[:, :-1]], dim=1
        )  # (B, L, 3, 3)
        trans_vel = x[..., 0:3]  # (B, L, 3)
        trans = torch.zeros_like(trans_vel)  # (B, L, 3)
        trans[:, 1:] = trans_vel[:, :-1]  # (B, L, 3)
        trans = trans[:, :, None]  # (B, L, 1, 3)
        trans = matrix.get_position_from_rotmat(trans, vel_global_orient_rotmat)  # (B, L, 1, 3)
        trans = trans[:, :, 0]  # (B, L, 3)
        trans = torch.cumsum(trans, dim=1)  # (B, L, 3)
        trans = trans + trans0  # (B, L, 3)

        smpl_rot6d = torch.zeros_like(x[..., 72:198])
        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)

        root_pos = joints[:, :, :1]  # (B, L, 1, 3)

        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)

        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root_pos  # (B, L, J-1, 3)
        joints = torch.cat([root_pos, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def get_fk_local_joints_from_x(self, x):
        """
        get forward kinematics local joints from local pose
        Args:
            x: (B, L, C=202)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        smpl_rot6d = x[..., 72:198]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        local_joints = self.forward_func(smplpose)  # (B, L, J, 3)
        # SMPL FK has a root offset
        local_joints = local_joints - local_joints[:, :, :1]  # (B, L, J, 3)
        return local_joints

    def get_fk_joints_from_x(self, x, gt_smplpose=None):  # TODO: change gt_smplpose to trans and global_orients
        """
        get forward kinematics joints from local pose and gt_transl, global_orient
        Args:
            x: (B, L, C=202)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """
        B, L, _ = x.shape
        smpl_rot6d = x[..., 72:198]
        trans = torch.zeros((B, L, 3), device=x.device)
        global_orient_rot6d = torch.zeros((B, L, 6), device=x.device)
        if gt_smplpose is not None:
            trans = gt_smplpose[..., :3]
            global_orient_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(gt_smplpose[..., 3:6]))

        smplrot6d = torch.cat([trans, global_orient_rot6d, smpl_rot6d], dim=-1)  # (B, L, 135)
        smplpose = self.smplrot6d2smplpose(smplrot6d)  # (B, L, 69)
        joints = self.forward_func(smplpose)  # (B, L, J, 3)
        return joints

    def get_fk_local_joints_from_local(self, x):
        """
        get forward kinematics local joints from local joints
        Args:
            x: (B, L, C=202)
        Returns:
            local_joints: (B, L, J, 3) without global trans and global orient
        """

        B, L, _ = x.shape
        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        local_joints = torch.cat([torch.zeros_like(local_joints[..., :1, :]), local_joints], dim=-2)  # (B, L, J, 3)
        return local_joints

    def get_fk_joints_from_local(self, x, gt_smplpose):
        """
        get forward kinematics global joints from local joints and gt transl, global_orient
        Args:
            x: (B, L, C=202)
            gt_smplpose: (B, L, 69)
        Returns:
            joints: (B, L, J, 3)
        """

        B, L, _ = x.shape
        gt_joints = self.forward_func(gt_smplpose)  # (B, L, J, 3)
        root = gt_joints[:, :, :1]  # (B, L, 1, 3)
        local_joints = x[..., 9:72]  # (B, L, 21*3)
        local_joints = local_joints.reshape(B, L, -1, 3)
        global_orient = gt_smplpose[..., 3:6]
        global_orient_rotmat = axis_angle_to_matrix(global_orient)  # (B, L, 3)
        local_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        local_joints = local_joints + root
        joints = torch.cat([root, local_joints], dim=-2)  # (B, L, J, 3)

        return joints

    def get_noisy_meanstd(self):

        noisyobs_mean = self.mean[72:198]
        noisyobs_std = self.std[72:198]
        return noisyobs_mean, noisyobs_std


# V3 with beta
class SMPLRelVecV4EnDecoder(SMPLRelVecV3EnDecoder):

    def encode(self, data, length=None):
        """
        Args:
            data: (B, L, c=69)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=212, L)
        """
        data = data.clone()
        B, L, _ = data.shape
        ayfz_smplpose, ayfz_joints = self.get_ayfz_smplpose(data)  # (B, L, c), (B, L, J, 3)
        trans = ayfz_smplpose[..., :3]  # (B, L, 3)
        global_orient = ayfz_smplpose[..., 3:6]  # (B, L, 3)
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        trans_vel = self.get_trans_vel(trans, global_orient_mat)  # (B, L, 3)
        local_joints = ayfz_joints - ayfz_joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)

        ayfz_smplrot6d = self.smplpose2smplrot6d(ayfz_smplpose)  # (B, L, C)
        global_orient_rot6d = matrix_to_rotation_6d(global_orient_mat)  # (B, L, 6)

        contact_label = self.detect_foot_contact(ayfz_joints)  # (B, L, 4)

        beta = self.beta[:, None].expand(B, L, -1)  # (B, L, 10)
        beta = beta.to(data.device)

        x = torch.cat(
            [
                trans_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 9:72
                ayfz_smplrot6d[..., 9:],  # (B, L, (J-1)*6) -> 72:198
                contact_label,  # (B, L, 4) -> 198:202
                beta,  # (B, L, 10) -> 202:212
            ],
            dim=-1,
        )  # (B, L, 212)

        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def decode_beta(self, x):
        """
        decode beta from x
        Args:
            x: (B, C=212, L)
        Returns:
            beta: (B, L, 10)
        """
        x_ori = self.denorm(x[:, :212])  # (B, L, 212)
        beta = x_ori[..., 202:212]
        return beta


# V4 with funcs for getting global_orient and global_transl in camera coordinate
class SMPLRelVecV5EnDecoder(SMPLRelVecV4EnDecoder):
    def __init__(self, stats_module, stats_name, stats_incam_name, forward_func="fk", noise_pose_k=0):
        super().__init__(stats_module, stats_name, forward_func, noise_pose_k)
        try:
            stats = getattr(importlib.import_module(stats_module), stats_incam_name)
            Log.info(f"We use {stats_incam_name} for pose in camera statistics!")
            self.register_buffer("poseincam_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("poseincam_std", torch.tensor(stats["std"]).float(), False)
        except Exception as e:
            print(e)
            Log.info(f"Cannot find {stats_incam_name} in {stats_module}! Use zero as mean, one as std!")
            self.register_buffer("poseincam_mean", torch.zeros(1).float(), False)
            self.register_buffer("poseincam_std", torch.ones(1).float(), False)

    def encode_pose_incam(self, data, length=None):
        """
        Args:
            data: (B, L, c=6) transl and global_orient in camera coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=9, L)
        """
        data = data.clone()
        B, L, _ = data.shape
        trans_incam = data[..., :3]  # (B, L, 3)
        global_orient_incam_aa = data[..., 3:6]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)
        x = torch.cat(
            [
                trans_incam,  # (B, L, 3) -> 0:3
                global_orient_incam_rot6d,  # (B, L, 6) -> 3:9
            ],
            dim=-1,
        )  # (B, L, 9)
        x = (x - self.poseincam_mean) / self.poseincam_std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def decode_pose_incam(self, x):
        """
        Args:
            x: (B, C=9, L)
        Returns:
            param: (B, L, c=6), transl and global_orient in camera coordinate
        """
        x = x.permute(0, 2, 1)  # (B, L, 9)
        x = x * self.poseincam_std + self.poseincam_mean  # (B, L, 9)
        trans_incam = x[..., :3]  # (B, L, 3)
        global_orient_incam_rot6d = x[..., 3:9]  # (B, L, 6)
        global_orient_incam_aa = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_incam_rot6d))  # (B, L, 3)
        param_incam = torch.cat([trans_incam, global_orient_incam_aa], dim=-1)  # (B, L, 6)

        return param_incam


# V5 only global_orient in camera coordinate
class SMPLRelVecV51EnDecoder(SMPLRelVecV5EnDecoder):

    def encode_pose_incam(self, data, length=None):
        """
        Args:
            data: (B, L, c=6) global_orient in camera coordinate
            length: (B,), effective length of each sample
        Returns:
            x: (B, C=6, L)
        """
        data = data.clone()
        B, L, _ = data.shape
        global_orient_incam_aa = data[..., 3:6]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)
        x = torch.cat(
            [
                global_orient_incam_rot6d,  # (B, L, 6) -> 0:6
            ],
            dim=-1,
        )  # (B, L, 6)
        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x = (x - poseincam_mean) / poseincam_std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.set_default_padding(x, length)
        return x

    def decode_pose_incam(self, x):
        """
        Args:
            x: (B, C=6, L)
        Returns:
            param: (B, L, c=3), global_orient in camera coordinate
        """
        x = x.permute(0, 2, 1)  # (B, L, 9)
        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x = x * poseincam_std + poseincam_mean
        global_orient_incam_rot6d = x[..., :6]  # (B, L, 6)
        global_orient_incam_aa = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_incam_rot6d))  # (B, L, 3)

        return global_orient_incam_aa
