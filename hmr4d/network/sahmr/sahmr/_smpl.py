"""
This file contains the definition of the SMPL model

It is adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/)
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import pickle
from pytorch3d.transforms import axis_angle_to_matrix
import hmr4d.network.sahmr.sahmr.data.config as cfg


class SMPL(nn.Module):
    def __init__(self, gender="neutral"):
        super(SMPL, self).__init__()

        if gender == "m":
            model_file = cfg.SMPL_Male
        elif gender == "f":
            model_file = cfg.SMPL_Female
        else:
            model_file = cfg.SMPL_FILE

        smpl_model = pickle.load(open(model_file, "rb"), encoding="latin1")
        J_regressor = smpl_model["J_regressor"].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data  # (236,)
        i = torch.LongTensor(np.array([row, col]))
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer("J_regressor", torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense())
        self.register_buffer("weights", torch.FloatTensor(smpl_model["weights"]))  # ?
        self.register_buffer("posedirs", torch.FloatTensor(smpl_model["posedirs"]))
        self.register_buffer("v_template", torch.FloatTensor(smpl_model["v_template"]))
        self.register_buffer("shapedirs", torch.FloatTensor(np.array(smpl_model["shapedirs"])))
        self.register_buffer("faces", torch.from_numpy(smpl_model["f"].astype(np.int64)))
        self.register_buffer("kintree_table", torch.from_numpy(smpl_model["kintree_table"].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer(
            "parent",
            torch.LongTensor(
                [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]
            ),
        )

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        # J_regressor_extra = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
        # self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = cfg.JOINTS_IDX

        J_regressor_h36m_correct = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M_correct)).float()
        self.register_buffer("J_regressor_h36m_correct", J_regressor_h36m_correct)

        self.H36M_J17_PELVIS_ID = cfg.H36M_J17_NAME.index("Pelvis")
        self.H36M_J17_TO_J14 = cfg.H36M_J17_TO_J14
        # selected extra verts as joints for smpl and smplh
        # fmt: off
        self.select_verts = [332, 6260, 2800, 4071, 583, 3216, 3226, 3387, 6617, 6624, 6787, 2746, 2319, 2445, 2556, 2673, 6191, 5782, 5905, 6016, 6133]
        self.smpl2op = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        # fmt: on

    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = axis_angle_to_matrix(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = (
            torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1))
            .view(6890, batch_size, 4, 4)
            .transpose(0, 1)
        )
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor])
        joints_extra = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

    def get_op_joints(self, vertices):
        """
        This method is used to get the openpose joint locations
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 25, 3)
        """
        joints = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor])
        joints = torch.cat((joints, vertices[:, self.select_verts]), dim=1)
        joints = joints[:, self.smpl2op]
        return joints

    def get_h36m_joints(self, vertices):
        """vertices (B, 6890, 3) -> H36M joints (B, 17, 3)"""
        return torch.einsum("bik,ji->bjk", [vertices, self.J_regressor_h36m_correct])

    def get_h36m_joints14(self, vertices):
        """vertices (B, 6890, 3) -> H36M joints14 (B, 14, 3)"""
        joints = self.get_h36m_joints(vertices)
        joints14 = joints[:, self.H36M_J17_TO_J14, :]
        return joints14

    def get_h36m_pelvis(self, vertices):
        """vertices (B, 6890, 3) -> H36M joints14 (B, 1, 3)"""
        joints = self.get_h36m_joints(vertices)
        return joints[:, [self.H36M_J17_PELVIS_ID], :]

    def get_r_h36m_verts_joints14(self, vertices):
        """
        Get regularized(-pelvis) H36M verts and joints14
        Input:
            vertices: (B, 6890, 3)
        Output:
            vertices: (B, 6890, 3)
            joints14: (B, 14, 3)
        """
        joints = self.get_h36m_joints(vertices)
        pelvis = joints[:, [self.H36M_J17_PELVIS_ID], :]
        joints14 = joints[:, self.H36M_J17_TO_J14, :]
        regu_verts = vertices - pelvis
        regu_joints14 = joints14 - pelvis
        return regu_verts, regu_joints14

    def initialize_h36m_verts_and_joints14(self):
        """
        1. if coord is
            cr: upside down global orient
            azr: up is +z
        2. r h36m verts+joints14
        """
        # 1. upside down global orient
        init_pose = torch.zeros((1, 72))
        init_pose[:, 0] = 3.1416  # Rectify "upside down" reference mesh in global coord
        init_betas = torch.zeros((1, 10))
        init_verts = self.forward(init_pose, init_betas)

        # from hmr4d.utils.vis3d_utils import make_vis3d
        # vis3d = make_vis3d(None, "default_init_smpl", time_postfix=True)
        # vis3d.add_point_cloud(init_verts[0], name='pi00')

        # 2. r h36m verts+joints14
        init_verts, init_joints14 = self.get_r_h36m_verts_joints14(init_verts)
        return init_verts, init_joints14


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (sparse,) = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def scipy_to_pytorch(M):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptM = []
    for i in range(len(M)):
        u = scipy.sparse.coo_matrix(M[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptM.append(torch.sparse_coo_tensor(i, v, u.shape))
    return ptM


# def adjmat_sparse(adjmat, nsize=1):
#     """Create row-normalized sparse graph adjacency matrix."""
#     adjmat = scipy.sparse.csr_matrix(adjmat)
#     if nsize > 1:
#         orig_adjmat = adjmat.copy()
#         for _ in range(1, nsize):
#             adjmat = adjmat * orig_adjmat
#     adjmat.data = np.ones_like(adjmat.data)  # 本来不是1，现在改成1
#     adjmat = adjmat.toarray()
#     for i in range(adjmat.shape[0]):  # 对角线也改成1
#         adjmat[i, i] = 1
#     num_neighbors = np.array(1 / adjmat.sum(axis=-1))
#     adjmat = adjmat * num_neighbors
#     i = torch.LongTensor(np.argwhere(adjmat).T)
#     v = torch.from_numpy(adjmat[adjmat != 0]).float()
#     adjmat = torch.sparse_coo_tensor(i, v, adjmat.shape)
#     return adjmat


# def get_graph_params(filename, nsize=1):
#     """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
#     data = np.load(filename, encoding='latin1', allow_pickle=True)
#     A = data['A']
#     U = data['U']
#     D = data['D']
#     U, D = scipy_to_pytorch(A, U, D)
#     A = [adjmat_sparse(a, nsize=nsize) for a in A]
#     return A, U, D


class MeshSampler(nn.Module):
    """Mesh object that is used for handling certain graph operations."""

    def __init__(self, filename=cfg.SMPL_sampling_matrix, num_downsampling=1, nsize=1):
        super().__init__()
        self.num_downsampling = num_downsampling

        # load downsampling and upsampling weights
        # _A, _U, _D = get_graph_params(filename=filename, nsize=nsize)
        data = np.load(filename, encoding="latin1", allow_pickle=True)
        U = scipy_to_pytorch(data["U"])
        self._U_0 = U[0]  # self.register_buffer(f"_U_0", U[0], False)
        self._U_1 = U[1]

        D = scipy_to_pytorch(data["D"])
        self._D_0 = D[0]
        self._D_1 = D[1]

    def to(self, deivce: torch.device):
        # It seems that a sparse_coo_tensor do not goes to device automatically
        self._U_0 = self._U_0.to(deivce)
        self._U_1 = self._U_1.to(deivce)
        self._D_0 = self._D_0.to(deivce)
        self._D_1 = self._D_1.to(deivce)
        return self

    # @property
    # def adjmat(self):
    #     """Return the graph adjacency matrix at the specified subsampling level."""
    #     return self._A[self.num_downsampling].float()

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh.
        x: (B, N, 3)
        """
        if n2 is None:
            n2 = self.num_downsampling  # 默认的下采样目标=1
        if x.ndimension() < 3:  # 如果不是batch
            for i in range(n1, n2):
                x = spmm(getattr(self, f"_D_{i}"), x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(getattr(self, f"_D_{j}"), y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(getattr(self, f"_U_{i}"), x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(getattr(self, f"_U_{j}"), y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def downsample_to_verts431(self, init_verts):
        init_verts1723 = self.downsample(init_verts)
        init_verts431 = self.downsample(init_verts1723, n1=1, n2=2)  # (1, 431, 3)
        return init_verts431
