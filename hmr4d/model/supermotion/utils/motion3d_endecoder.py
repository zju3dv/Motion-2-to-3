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

    def __init__(self, stats_module, stats_name, forward_func="fk", default_smpl="smplx"):
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
        self._build_smpl()
        self.default_smpl = default_smpl

    def _build_smpl(self):
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
        skeleton = smplx_model["neutral"]().joints[0, :22]  # 22, 3
        parents = smplx_model["neutral"].bm.parents[:22]  # 22 int
        local_skeleton = skeleton - skeleton[parents]
        local_skeleton = torch.cat([skeleton[:1], local_skeleton[1:]])  # 22, 3
        local_skeleton = local_skeleton[None]  # 1, 22, 3
        betas = torch.zeros(1, 10)  # 1, 10
        self.register_buffer("betas", betas, False)
        self.gender = ["neutral"]
        self.modeltype = ["smplx"]
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
        self.register_buffer("skeleton", skeleton, False)

    def set_betas(self, betas):
        """
        Args:
            betas: (B, 10)
        """
        self.register_buffer("betas", betas, False)

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

    def smpl_forward(self, transl, global_orient, body_pose, betas=None, **kwargs):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
            betas: (B, L, 10)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = transl.shape
        if betas is None:
            betas = self.betas.clone()[:, None].expand(B, L, -1)
        else:
            if len(betas.shape) < 2:
                betas = betas[None, :].expand(B, -1)  # (B, 10)
            if len(betas.shape) < 3:
                betas = betas[:, None].expand(B, L, -1)  # (B, L, 10)
        betas = betas.to(transl.device)
        smpl_model = self.body_model[self.default_smpl]["neutral"]
        smpl_model = smpl_model.to(transl.device)
        smpl_out = smpl_model(
            betas=betas.reshape(-1, 10),
            transl=transl.reshape(-1, 3),
            global_orient=global_orient.reshape(-1, 3),
            body_pose=body_pose.reshape(-1, 63),
        )
        joints = smpl_out.joints[:, :22]  # (B*L, J, 3)
        joints = joints.reshape(B, L, -1, 3)
        verts = smpl_out.vertices  # (B*L, V, 3)
        verts = verts.reshape(B, L, -1, 3)
        return joints, verts

    def smpl_forward_varioustype(self, transl, global_orient, body_pose, betas=None, **kwargs):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
            betas: (B, L, 10)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = transl.shape
        if betas is None:
            betas = self.betas.clone()[:, None].expand(B, L, -1)
        betas = betas.to(transl.device)
        all_joints = []
        all_verts = []
        for i in range(B):
            if i >= len(self.modeltype):
                modeltype = self.modeltype[-1]
                gender = self.gender[-1]
            else:
                modeltype = self.modeltype[i]
                gender = self.gender[i]
            smpl_model = self.body_model[modeltype][gender].to(transl.device)

            betas_ = betas[i]
            transl_ = transl[i]
            global_orient_ = global_orient[i]
            body_pose_ = body_pose[i]
            smpl_out = smpl_model(
                betas=betas_.reshape(-1, 10),
                transl=transl_.reshape(-1, 3),
                global_orient=global_orient_.reshape(-1, 3),
                body_pose=body_pose_.reshape(-1, 63),
            )
            joints = smpl_out.joints[:, :22]  # (L, J, 3)
            verts = smpl_out.vertices  # (L, V, 3)
            all_joints.append(joints)
            all_verts.append(verts)
        all_joints = torch.stack(all_joints, dim=0)
        all_verts = torch.stack(all_verts, dim=0)
        return all_joints, all_verts

    def get_cr_verts(self, transl, global_orient, body_pose, betas=None, **kwargs):
        if betas is not None:
            self.set_betas(betas)
        joints, verts = self.smpl_forward(transl, global_orient, body_pose)
        # (B, L, J, 3), (B, L, V, 3)

        root = joints[:, :, :1]  # (B, L, 1, 3)
        cr_j3d = joints - root  # (B, L, J, 3)
        cr_verts = verts - root  # (B, L, V, 3)

        return cr_j3d, cr_verts

    @torch.no_grad()
    def get_cr_verts_wograd(self, transl, global_orient, body_pose, betas=None, **kwargs):
        if betas is not None:
            self.set_betas(betas)
        joints, verts = self.smpl_forward(transl, global_orient, body_pose)
        # (B, L, J, 3), (B, L, V, 3)

        root = joints[:, :, :1]  # (B, L, 1, 3)
        cr_j3d = joints - root  # (B, L, J, 3)
        cr_verts = verts - root  # (B, L, V, 3)

        return cr_j3d, cr_verts

    def fk_v2(self, body_pose, betas=None, global_orient=None, transl=None):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        if betas is None:
            local_skeleton = self.local_skeleton[:, None].expand(B, L, -1, -1).clone()  # (B, L, 22, 3)
            local_skeleton = local_skeleton.to(body_pose.device)
        else:
            smpl_model = self.body_model[self.default_smpl]["neutral"]
            smpl_model = smpl_model.to(body_pose.device)
            skeleton = smpl_model.get_skeleton(betas)  # (B, L, 22, 3)
            local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
            local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3

        return joints

    def fk_forward(self, transl, global_orient, body_pose, betas=None, **kwargs):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
            betas: (B, L, 10)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = transl.shape
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)
        if betas is None:
            local_skeleton = self.local_skeleton[:, None].expand(B, L, -1, -1).clone()  # (B, L, 22, 3)
            local_skeleton = local_skeleton.to(body_pose.device)
        else:
            if len(betas.shape) < 2:
                betas = betas[None, :].expand(B, -1)  # (B, 10)
            if len(betas.shape) < 3:
                betas = betas[:, None].expand(B, L, -1)  # (B, L, 10)
            smpl_model = self.body_model[self.default_smpl]["neutral"]
            smpl_model = smpl_model.to(body_pose.device)
            skeleton = smpl_model.get_skeleton(betas)  # (B, L, 22, 3)
            local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
            local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        local_skeleton[..., 0, :] += transl  # B, L, 22, 3
        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3

        return joints, None

    def get_ayfz_joints(self, transl, global_orient, body_pose):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
        Returns:
            ayfz_joints: (B, L, J, 3)
        """
        B, L, _ = transl.shape
        joints, _ = self.forward_func(transl, global_orient, body_pose)  # (B, L, 22, 3)
        if False:  # Visualize
            bid = 0
            gt_params = {
                "betas": self.betas[[bid]].expand(L, -1),  # (L, 10)
                "transl": transl,  # (L, 3)
                "global_orient": global_orient,  # (L, 3)
                "body_pose": body_pose,  # (L, 63)
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
        # put on the floor
        ayfz_joints_floor = ayfz_joints.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
        ayfz_joints[..., 1] = ayfz_joints[..., 1] - ayfz_joints_floor[:, None, None]
        return ayfz_joints

    def get_ayfz_smplpose(self, transl, global_orient, body_pose, betas=None, **kwargs):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
        Returns:
            ayfz_transl: (B, L, 3)
            ayfz_global_orient: (B, L, 3)
            body_pose: (B, L, 63)
            ayfz_joints: (B, L, J, 3)
        """
        B, L, _ = transl.shape

        joints, _ = self.forward_func(transl, global_orient, body_pose, betas)  # (B, L, 22, 3)
        joints_firstframe = joints[:, 0]  # (B, 22, 3)
        T_ay2ayfz = compute_T_ayfz2ay(joints_firstframe, inverse=True)  # (B, 4, 4)
        ayfz_joints = apply_T_on_points(joints.reshape(B, -1, 3), T_ay2ayfz).reshape(B, L, -1, 3)  # B, L, 22, 3
        ayfz_joints_floor = ayfz_joints.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
        ayfz_joints[..., 1] = ayfz_joints[..., 1] - ayfz_joints_floor[:, None, None]

        # ay to ayfz transl

        # smpl has root offset, should also be transformed
        root_offset = self.local_skeleton[:, :1]  # (B, 1, 3)
        root_offset = root_offset.to(transl.device)
        root_offset = root_offset.expand(B, L, -1).clone()  # (B, L, 3)
        transl_with_offset = transl + root_offset  # (B, L, 3)
        ayfz_transl_with_offset = matrix.get_position_from(transl_with_offset, T_ay2ayfz)  # (B, L, 3)
        ayfz_transl = ayfz_transl_with_offset - root_offset  # (B, L, 3)

        # put on the floor
        ayfz_transl[..., 1] = ayfz_transl[..., 1] - ayfz_joints_floor[:, None]

        # ay to ayfz rot
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        ayfz_global_orient_mat = matrix.get_mat_BfromA(T_ay2ayfz[:, None, :3, :3], global_orient_mat)
        ayfz_global_orient = matrix_to_axis_angle(ayfz_global_orient_mat)
        return ayfz_transl, ayfz_global_orient, body_pose, ayfz_joints

    def smplpose2smplrot6d(self, global_orient, body_pose):
        """
        Args:
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
        Returns:
            global_orient_rot6d: (B, L, 6)
            body_pose_rot6d: (B, L, 126)
        """
        B, L, _ = global_orient.shape
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)
        rot6d = matrix_to_rotation_6d(rotmat)  # (B, L, 22, 6)
        rot6d = rot6d.reshape(B, L, -1)  # (B, L, 132)
        global_orient_rot6d = rot6d[..., :6]  # (B, L, 6)
        body_pose_rot6d = rot6d[..., 6:]  # (B, L, 126)
        return global_orient_rot6d, body_pose_rot6d

    def smplrot6d2smplpose(self, global_orient_rot6d, body_pose_rot6d):
        """
        Args:
            global_orient_rot6d: (B, L, 6)
            body_pose_rot6d: (B, L, 126)
        Returns:
            global_orient: (B, L, 3)
            body_pose: (B, L, 63)
        """
        B, L, _ = global_orient_rot6d.shape
        rot6d = torch.cat([global_orient_rot6d, body_pose_rot6d], dim=-1)  # (B, L, 132)
        rot6d = rot6d.reshape(B, L, -1, 6)  # (B, L, 22, 6)
        rotmat = rotation_6d_to_matrix(rot6d)  # (B, L, 22, 3, 3)
        aa = matrix_to_axis_angle(rotmat)  # (B, L, 22, 3)
        aa = aa.reshape(B, L, -1)  # (B, L, 66)
        global_orient = aa[..., :3]
        body_pose = aa[..., 3:]
        return global_orient, body_pose

    def set_default_padding(self, x, length=None):
        """
        Args:
            x: (B, C, L)
            length: (B,), effective length of each sample
        Returns:
            x: (B, C, L)
        """
        if length is not None:
            for i, l in enumerate(length):
                x[i, :, l:] = 0.0
        return x

    def set_cfg(self, data):
        """
        Args:
            data (dict):
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
                skeleton: (B, J, 3)
                gender: (B), default=None
                modeltype: (B), default=None
                length: (B), default=None
        """
        B = data["transl"].shape[0]
        if "betas" not in data.keys() or data["betas"] == None:
            pass
        else:
            self.set_betas(data["betas"])

        if "skeleton" not in data.keys() or data["skeleton"] == None:
            pass
        else:
            self.set_skeleton(data["skeleton"])

        if "gender" not in data.keys() or data["gender"] == None:
            gender = ["neutral" for _ in range(B)]
        else:
            gender = data["gender"]
        self.set_gender(gender)

        if "modeltype" not in data.keys() or data["modeltype"] == None:
            modeltype = [self.default_smpl for _ in range(B)]
        else:
            modeltype = data["modeltype"]
        self.set_modeltype(modeltype)
        self.clear_ayfz()

    def convert2ayfzdata(self, data):
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

        ayfz_transl, ayfz_global_orient, body_pose, ayfz_joints = self.get_ayfz_smplpose(
            transl, global_orient, body_pose, betas
        )
        # (B, L, 3), (B, L, 3)         , (B, L, 63), (B, L, J, 3)

        # Save ayfz data
        self.ayfz_transl = ayfz_transl.clone()
        self.ayfz_global_orient = ayfz_global_orient.clone()
        self.body_pose = body_pose.clone()
        self.ayfz_joints = ayfz_joints

        output = {}
        for k, v in data.items():
            output[k] = v
        output["transl"] = ayfz_transl
        output["global_orient"] = ayfz_global_orient
        return output

    def clear_ayfz(self):
        self.ayfz_transl = None
        self.ayfz_global_orient = None
        self.body_pose = None
        self.ayfz_joints = None

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
        Returns:
            x: (B, C=145, L)
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

        x = torch.cat(
            [
                transl,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 9:135
                betas,  # (B, L, 10) -> 135:145
            ],
            dim=-1,
        )  # (B, L, 145)
        x = self.norm(x)  # (B, 145, L)

        # length = data.get("length", None)
        # x = self.set_default_padding(x, length)
        return x

    def decode(self, x):
        """
        Args:
            x: (B, C=135, L)
        Returns:
            output (dict): in ayfz coordinate
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
        """
        denorm_x = self.denorm(x)  # (B, L, 135)

        ayfz_transl = denorm_x[:, :, :3]  # (B, L, 3)
        ayfz_global_orient_rot6d = denorm_x[:, :, 3:9]  # (B, L, 6)
        body_pose_rot6d = denorm_x[:, :, 9:135]  # (B, L, 126)
        betas = denorm_x[:, :, 135:145]  # (B, L, 10)
        ayfz_global_orient, body_pose = self.smplrot6d2smplpose(ayfz_global_orient_rot6d, body_pose_rot6d)
        # (B, L, 3)       , (B, L, 63)
        output = {
            "transl": ayfz_transl,
            "global_orient": ayfz_global_orient,
            "body_pose": body_pose,
            "betas": betas,
        }

        return output

    def get_ayfz_data(self):
        try:
            data = {
                "transl": self.ayfz_transl,
                "global_orient": self.ayfz_global_orient,
                "body_pose": self.body_pose,
                "betas": self.betas.clone(),
                "skeleton": self.skeleton.clone(),
                "gender": self.gender,
                "modeltype": self.modeltype,
                "joints": self.ayfz_joints,
            }
        except AttributeError:
            print("WARNING: This function requires run func 'convert2ayfzdata' first! So return None instead!")
            data = None
        return data

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

    def norm(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            x: (B, C, L)
        """
        x = (x - self.mean) / self.std  # (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        return x

    def denorm(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            x: (B, L, C)
        """
        x = x.permute(0, 2, 1)  # (B, L, 273)
        x = x * self.std + self.mean  # (B, L, 273)
        return x


# transl_vel + global_orient + body_pose + local_joints + betas + foot_contact
class SMPLRelVecV50EnDecoder(SMPLEnDecoder):
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

    def localjoints_forward(self, transl, global_orient, local_joints, **kwargs):
        """
        Args:
            transl: (B, L, 3)
            global_orient: (B, L, 3)
            local_joints: (B, L, 63) 63=(J-1) * 3
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L, _ = transl.shape
        body_pose = torch.zeros((B, L, 63), device=transl.device)
        joints, _ = self.forward_func(transl, global_orient, body_pose)  # (B, L, J, 3)
        root_pos = joints[:, :, :1]  # (B, L, 1, 3)

        global_orient_rotmat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        local_joints = local_joints.reshape(B, L, -1, 3)  # (B, L, J-1, 3)
        global_joints = matrix.get_position_from_rotmat(local_joints, global_orient_rotmat)  # (B, L, J-1, 3)
        global_joints = global_joints + root_pos  # (B, L, J-1, 3)
        joints = torch.cat([root_pos, global_joints], dim=-2)  # (B, L, J, 3)

        return joints, None

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
        Returns:
            x: (B, C=212, L)
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

        # relative to root
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        local_joints = joints - joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)

        contact_label = self.detect_foot_contact(joints)  # (B, L, 4)

        x = torch.cat(
            [
                transl_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 9:72
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 72:198
                contact_label,  # (B, L, 4) -> 198:202
                betas,  # (B, L, 10) -> 202:212
            ],
            dim=-1,
        )  # (B, L, 212)

        x = self.norm(x)

        # length = data.get("length", None)
        # x = self.set_default_padding(x, length)
        return x

    def decode(self, x):
        """
        deocde with gt first frame transl
        Args:
            x: (B, C=212, L)
        Returns:
            output (dict): in ayfz coordinate
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
            data: (B, L, c=69),
        """
        denorm_x = self.denorm(x)  # (B, L, C)
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

        body_pose_rot6d = denorm_x[..., 72:198]
        ayfz_global_orient, body_pose = self.smplrot6d2smplpose(ayfz_global_orient_rot6d, body_pose_rot6d)
        # (B, L, 3)       , (B, L, 63)

        local_joints = denorm_x[..., 9:72].reshape(B, L, -1, 3)  # (B, L, J-1, 3)
        contact_label = denorm_x[..., 198:202]  # (B, L, 4)

        betas = denorm_x[:, :, 202:212]  # (B, L, 10)
        output = {
            "transl": ayfz_transl,
            "global_orient": ayfz_global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "local_joints": local_joints,
            "contact_label": contact_label,
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
        noisyobs_mean = self.mean[72:198]
        noisyobs_std = self.std[72:198]
        return noisyobs_mean, noisyobs_std


# V50 + global_orient in camera coordinate
class SMPLRelVecV51EnDecoder(SMPLRelVecV50EnDecoder):
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
                transl_incam: (B, L, 3)
                "global_orient_incam": (B, L, 3)
        Returns:
            x: (B, C=218, L)
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

        # relative to root
        global_orient_mat = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
        local_joints = joints - joints[:, :, :1]  # (B, L, J, 3)
        local_joints = matrix.get_relative_direction_to(local_joints, global_orient_mat)  # (B, L, J, 3)

        contact_label = self.detect_foot_contact(joints)  # (B, L, 4)

        x = torch.cat(
            [
                transl_vel,  # (B, L, 3) -> 0:3
                global_orient_rot6d,  # (B, L, 6) -> 3:9
                local_joints[..., 1:, :].flatten(2),  # (B, L, (J-1)*3) -> 9:72
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 72:198
                contact_label,  # (B, L, 4) -> 198:202
                betas,  # (B, L, 10) -> 202:212
            ],
            dim=-1,
        )  # (B, L, 212)

        x = self.norm(x)  # (B, 212, L)

        # length = data.get("length", None)
        # x = self.set_default_padding(x, length)  # (B, 212, L)

        global_orient_incam_aa = data["global_orient_incam"]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)

        x_incam = torch.cat(
            [
                global_orient_incam_rot6d,  # (B, L, 6) -> 0:6 / 212:218
            ],
            dim=-1,
        )  # (B, L, 6)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = (x_incam - poseincam_mean) / poseincam_std  # (B, L, C)
        x_incam = x_incam.permute(0, 2, 1)  # (B, C, L)

        # length = data.get("length", None)
        # x_incam = self.set_default_padding(x_incam, length)

        x = torch.cat([x, x_incam], dim=1)  # (B, 218, L)

        return x

    def decode(self, x):
        """
        deocde with gt first frame transl
        Args:
            x: (B, C=218, L)
        Returns:
            output (dict): in ayfz coordinate
                transl: (B, L, 3)
                global_orient: (B, L, 3)
                body_pose: (B, L, 63)
                betas: (B, 10)
                "global_orient_incam": (B, L, 3)
        """
        denorm_x = self.denorm(x[:, :212])  # (B, L, C)
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
            pass
            # print("No gt_ayfz_transl, use zero as first frame transl.")

        body_pose_rot6d = denorm_x[..., 72:198]
        ayfz_global_orient, body_pose = self.smplrot6d2smplpose(ayfz_global_orient_rot6d, body_pose_rot6d)
        # (B, L, 3)       , (B, L, 63)

        local_joints = denorm_x[..., 9:72].reshape(B, L, -1, 3)  # (B, L, J-1, 3)
        contact_label = denorm_x[..., 198:202]  # (B, L, 4)

        betas = denorm_x[:, :, 202:212]  # (B, L, 10)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = x[:, 212:218, :]  # (B, 6, L)
        x_incam = x_incam.permute(0, 2, 1)  # (B, L, 6)
        x_incam = x_incam * poseincam_std + poseincam_mean
        global_orient_incam_rot6d = x_incam[..., :6]  # (B, L, 6)
        global_orient_incam_aa = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_incam_rot6d))  # (B, L, 3)

        output = {
            "transl": ayfz_transl,
            "global_orient": ayfz_global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "local_joints": local_joints,
            "contact_label": contact_label,
            "global_orient_incam": global_orient_incam_aa,
        }

        return output

    def decode_raw_global_root(self, x, decode_dict):
        # raw velocity in SMPL coord
        s_transl_vel = self.denorm(x[:, :212])[..., 0:3]  # (B, L, 3), 0->L-1 are valid
        ayfz_global_orient = decode_dict["global_orient"]  # (B, L, 3)

        return s_transl_vel, ayfz_global_orient

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
                local_joints,  # (B, L, (J-1)*3) -> 9:72
                body_pose_rot6d,  # (B, L, (J-1)*6) -> 72:198
                data["contact_label"],  # (B, L, 4) -> 198:202
                betas,  # (B, L, 10) -> 202:212
            ],
            dim=-1,
        )  # (B, L, 212)

        x = self.norm(x)  # (B, 212, L)

        # Incam params in x (Please fix this)
        global_orient_incam_aa = data["global_orient_incam"]  # (B, L, 3)
        global_orient_incam_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_incam_aa))  # (B, L, 6)

        x_incam = torch.cat(
            [
                global_orient_incam_rot6d,  # (B, L, 6) -> 0:6 / 212:218
            ],
            dim=-1,
        )  # (B, L, 6)

        poseincam_mean = self.poseincam_mean[..., 3:9]
        poseincam_std = self.poseincam_std[..., 3:9]
        x_incam = (x_incam - poseincam_mean) / poseincam_std  # (B, L, C)
        x_incam = x_incam.permute(0, 2, 1)  # (B, C, L)

        x = torch.cat([x, x_incam], dim=1)  # (B, 218, L)

        return x
