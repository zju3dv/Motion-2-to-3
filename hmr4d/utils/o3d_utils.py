import numpy as np
import torch

try:
    import open3d as o3d
except:
    print("open3d not installed")
from .wis3d_utils import get_const_colors, color_schemes
from .skeleton_utils import SMPL_SKELETON, NBA_SKELETON, GYM_SKELETON, COLOR_NAMES
import hmr4d.utils.matrix as matrix
import smplx


STATE = {
    "play": True,
    "reset": False,
    "next": False,
    "back": False,
    "after": False,
    "prev": False,
    "iter_play": False,
}
VIS_STATE = {i + 1: True for i in range(10)}

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

cot2_viz_pastel_color_schemes = {
    0: ([255, 168, 154], [216, 140, 172]),
    1: ([183, 255, 191], [239, 203, 157]),
    2: ([183, 255, 255], [114, 183, 156]),
    3: ([183, 255, 255], [148, 173, 210]),
    4: ([255, 183, 255], [189, 152, 216]),
}


def get_worldcoordinate_line_set():
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [3, 0],
        [0, 4],
    ]
    colors = ["red", "green", "blue"]
    colors = np.array([color_schemes[c][1] for c in colors]) / 255.0
    w_coord_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    w_coord_line_set.colors = o3d.utility.Vector3dVector(colors)
    return w_coord_line_set


def get_coordinate_mesh(mat=None, size=1.0):
    # mat: (4, 4)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if mat is not None:
        mesh.transform(mat)
    return mesh


def get_ground_mesh(pos, upaxis="y"):
    """_summary_

    Args:
        pos (tensor): (progress, T, J, 3)
        upaxis (str, optional): upward axis name. Defaults to "y".
    """
    # 0.02 for foot height
    foot_thresh = 0.02
    if upaxis == "y":
        lowest_value = pos[-1][..., 1].min()
        translation = np.array([-5.0, lowest_value - 1.0 - foot_thresh, -5.0])
        mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=1, depth=10)
    elif upaxis == "z":
        lowest_value = pos[-1][..., 2].min()
        translation = np.array([-5.0, -5.0, lowest_value - 1.0 - foot_thresh])
        mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=1)

    else:
        lowest_value = pos[-1][..., 0].min()
        translation = np.array([lowest_value - 1.0 - foot_thresh, -5.0, -5.0])
        mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=10, depth=10)
    # relative = False, translate the center of the mesh to the target position
    # mesh.translate(translation, relative=False)
    mesh.translate(translation)
    return mesh


def add_camera_line_set(w2c):
    # w2c: (4, 4)
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = ["red", "green", "blue"]
    colors = np.array([color_schemes[c][1] for c in colors]) / 255.0
    cam_line_pts = matrix.get_position_from(np.array(points), np.linalg.inv(w2c))

    camera_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_line_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    camera_line_set.colors = o3d.utility.Vector3dVector(colors)
    return camera_line_set


def play_callback(vis, key, action):
    print("play", "key", key, "action", action)
    STATE["play"] = True


def stop_callback(vis, key, action):
    print("stop", "key", key, "action", action)
    STATE["play"] = False


def reset_callback(vis, key, action):
    print("reset", "key", key, "action", action)
    STATE["reset"] = True


def next_callback(vis, key, action):
    print("next", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["next"] = True


def back_callback(vis, key, action):
    print("back", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["back"] = True


def after_callback(vis, key, action):
    print("after", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["after"] = True


def prev_callback(vis, key, action):
    print("prev", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["prev"] = True


def help_callback(vis, key, action):
    print("help", "key", key, "action", action)
    if key == 1 or key == 2:
        print(
            "\n"
            "------------Help Guide------------------\n"
            "P: Play\n"
            "S: Stop\n"
            "R: Reset to time=0.\n"
            "N: Next frame.\n"
            "B: Previous frame.\n"
            "U: Next DDPM step.\n"
            "Y: Previous DDPM step.\n"
            "-----------------------------------------\n"
            "1: Enable/Disable generated 3D motions.\n"
            "2: Enable/Disable 2D motions in camera frame.\n"
            "3: Enable/Disable estimated 3D motions from Mocap in global coordinate.\n"
            "4: Enable/Disable GT 3D motions.\n"
            "5: Enable/Disable estimated 3D motions in camera frame.\n"
            "-----------------------------------------\n"
        )


def vis1_callback(vis, key, action):
    print("vis1", "key", key, "action", action)
    if key == 1:
        VIS_STATE[1] = not VIS_STATE[1]
        if VIS_STATE[1]:
            print("Show 1 - target 3d pos")
        else:
            print("Not Show 1 - target 3d pos")


def vis2_callback(vis, key, action):
    print("vis2", "key", key, "action", action)
    if key == 1:
        VIS_STATE[2] = not VIS_STATE[2]
        if VIS_STATE[2]:
            print("Show 2 - 2d pos")
        else:
            print("Not Show 2 - 2d pos")


def vis3_callback(vis, key, action):
    print("vis3", "key", key, "action", action)
    if key == 1:
        VIS_STATE[3] = not VIS_STATE[3]
        if VIS_STATE[3]:
            print("Show 3 - pred 3d pos")
        else:
            print("Not Show 3 - pred 3d pos")


def vis4_callback(vis, key, action):
    print("vis4", "key", key, "action", action)
    if key == 1:
        VIS_STATE[4] = not VIS_STATE[4]
        if VIS_STATE[4]:
            print("Show 4 - gt 3d pos")
        else:
            print("Not Show 4 - gt 3d pos")


def vis5_callback(vis, key, action):
    print("vis5", "key", key, "action", action)
    if key == 1:
        VIS_STATE[5] = not VIS_STATE[5]
        if VIS_STATE[5]:
            print("Show 5 - camera-view 3d pos")
        else:
            print("Not Show 5 - camera-view 3d pos")


def add_skeleton(pos, r=0.05, resolution=10, skeleton_type="smpl"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    else:
        raise NotImplementedError

    kinematic_chain = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["kinematic_chain"]
    ]
    J = pos.shape[0]

    color_names = COLOR_NAMES[: len(kinematic_chain)]
    m_colors = []
    bones = []
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        color_ = get_const_colors(color_name, partial_shape=(num_line,), alpha=1.0)
        m_colors.append(color_[..., :3].cpu().detach().numpy())
        bones.append(np.stack((np.array(chain)[:-1], np.array(chain)[1:]), axis=-1))
    m_colors = np.concatenate(m_colors, axis=0)
    bones = np.concatenate(bones, axis=0)

    color_names = COLOR_NAMES
    joints_category = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["joints_category"]
    ]
    joints_mesh = []
    joints_color = []
    for i in range(J):
        for j, joints_ in enumerate(joints_category):
            if i in joints_:
                joints_color.append(color_schemes[color_names[j]][1])
                break
    joints_color = np.array(joints_color) / 255.0
    for i in range(J):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(joints_color[i])
        joints_mesh.append(mesh)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pos)
    line_set.lines = o3d.utility.Vector2iVector(bones)
    line_set.colors = o3d.utility.Vector3dVector(m_colors)
    return joints_mesh, line_set


def add_line(pos_s, pos_e):
    color_names = [i % 5 for i in range(len(pos_s))]
    m_colors = np.array([cot2_viz_pastel_color_schemes[c][1] for c in color_names]) / 255.0
    bones = [[i, i + len(pos_s)] for i in range(len(pos_s))]

    pos = np.concatenate((pos_s, pos_e), axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pos)
    line_set.lines = o3d.utility.Vector2iVector(bones)
    line_set.colors = o3d.utility.Vector3dVector(m_colors)
    return line_set


def pos_2dto3d(pos_2d, w2c, z=1, is_pinhole=True):
    """_summary_
        given pos_2d in camera coordiante,
        return corresponding 3d pos in world coordinate

    Args:
        pos_2d: (p, T, 22, 2)
        w2c: (p, T, 4, 4)
        z (int, optional): (p, T, 3)
        is_pinhole (bool, optional): _description_. Defaults to True.

    Returns:
        _pos_3d: (p, T, 22, 3)
    """

    if isinstance(pos_2d, torch.Tensor):
        pos_2d = pos_2d.cpu().detach().numpy()
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.cpu().detach().numpy()
    if isinstance(z, torch.Tensor):
        z = z.cpu().detach().numpy()

    # we assume pinhole camera
    pos_2d_shape = pos_2d.shape  # proress, T, 22, 2
    if isinstance(z, np.ndarray):
        # (progress, T) -> (progress, T, J)
        z = np.concatenate([z[..., None]] * pos_2d_shape[-2], axis=-1)
        z = z.reshape(-1, 1)
    pos_2d = pos_2d.reshape(-1, 2)
    w2c = w2c.reshape(-1, 4, 4)
    if is_pinhole:
        pos_3d = np.concatenate((pos_2d * z, z * np.ones((pos_2d.shape[0], 1))), axis=-1)
    else:
        pos_3d = np.concatenate((pos_2d, z * np.ones((pos_2d.shape[0], 1))), axis=-1)
    pos_3d = pos_3d.reshape(-1, pos_2d_shape[-2], 3)
    pos_3d = matrix.get_position_from(pos_3d, np.linalg.inv(w2c))
    pos_3d = pos_3d.reshape(pos_2d_shape[:-1] + (3,))
    return pos_3d


def get_good_z_for_2dvis(w2c, root, is_pinhole=True):
    """_summary_
        given w2c and root of skeleton,
        return a good z for 2d pos visualiation (line for check triangulation)

    Args:
        w2c: (p, L, 4, 4)
        root: (p, L, 3)
        is_pinhole (bool, optional): _description_. Defaults to True.

    Returns:
        max_z: (p, L, 3)
        min_z: (p, L, 3)
    """
    if isinstance(w2c, torch.Tensor):
        w2c = w2c.detach().cpu().numpy()
    if isinstance(root, torch.Tensor):
        root = root.detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)  # p, L, 4, 4
    dist = np.linalg.norm(matrix.get_position(c2w) - root, axis=-1)  # p, L, 3
    if is_pinhole:
        min_z = dist * 0.5
        max_z = dist * 1.2
    else:
        min_z = dist * 0.25
        max_z = dist * 1.25
    return max_z, min_z


class o3d_skeleton_animation:
    def __init__(
        self,
        pos,
        pos_2d=None,
        w2c=None,
        pred_pos=None,
        gt_pos=None,
        c_pos=None,
        is_pinhole=False,
        name="",
        upaxis="y",
        skeleton_type="smpl",
    ):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions in world coordinate
            pos_2d (tensor): (progress, V, T, J, 2) 2d joints positions relative to each camera view
            pred_pos (tensor): (progress, N, T, J, 3) 3d joints positions directly predicted by other models.
            gt_pos (tensor): (progress, N, T, J, 3) groudtruth 3d joints positions.
            c_pos (tensor): (progress, V, T, J, 3) 3d joints positions directly predicted in camera views.
            w2c (tensor): (progress, T, V, 4, 4) world2camera matrix (inverse of camera transformation)
            is_pinhole (bool): if true, we use perspective, otherwise orthogonal projection
            name (str, optional): text description of this motion sequence. Defaults to "".
            upaxis (str, optional): upward axis name. Defaults to "y".
            skeleton_type (str, optional): "smpl" or "nba". Defaults to "smpl".
        """
        if len(pos.shape) == 3:
            pos = pos[None]
        self.pos = pos
        if pos_2d is not None and len(pos_2d.shape) == 4:
            p = pos.shape[0]
            pos_2d = pos_2d[None].expand(p, -1, -1, -1, -1)  # progress, V, T, J, 2
        self.pos_2d = pos_2d
        self.pred_pos = pred_pos
        if gt_pos is not None and len(gt_pos.shape) == 3:
            gt_pos = gt_pos[None]  # 1, T, J, 3
        if gt_pos is not None and len(gt_pos.shape) == 4:
            p = pos.shape[0]
            gt_pos = gt_pos[None].expand(p, -1, -1, -1, -1)  # progress, 1, T, J, 3
        self.gt_pos = gt_pos
        self.c_pos = c_pos
        if w2c is not None and len(w2c.shape) == 3:
            l = pos.shape[1]
            w2c = w2c[None].expand(l, -1, -1, -1)  # T, V, 4, 4
        if w2c is not None and len(w2c.shape) == 4:
            p = pos.shape[0]
            w2c = w2c[None].expand(p, -1, -1, -1, -1)  # progress, T, V, 4, 4
        self.w2c = w2c
        self.is_pinhole = is_pinhole

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_action_callback(ord("P"), play_callback)  # Play animation
        vis.register_key_action_callback(ord("S"), stop_callback)  # Stop playing
        vis.register_key_action_callback(ord("R"), reset_callback)  # Reset
        vis.register_key_action_callback(ord("N"), next_callback)  # Next
        vis.register_key_action_callback(ord("B"), back_callback)  # Previous
        vis.register_key_action_callback(ord("U"), after_callback)  # Next ddpm iter
        vis.register_key_action_callback(ord("Y"), prev_callback)  # Previous ddpm iter

        vis.register_key_action_callback(ord("H"), help_callback)  # Help

        vis.register_key_action_callback(ord("1"), vis1_callback)  # Whether vis pos
        vis.register_key_action_callback(ord("2"), vis2_callback)  # Whether vis pos_2d
        vis.register_key_action_callback(ord("3"), vis3_callback)  # Whether vis pred_pos
        vis.register_key_action_callback(ord("4"), vis4_callback)  # Whether vis gt_pos
        vis.register_key_action_callback(ord("5"), vis5_callback)  # Whether vis c_pos

        vis.create_window(name)
        self.vis = vis
        self.iter_i = pos.shape[0] - 1
        self.time_i = 0
        vis.add_geometry(get_worldcoordinate_line_set())
        vis.add_geometry(get_coordinate_mesh(size=1.0))
        self._setup_3d_skeleton(pos, upaxis, skeleton_type)
        if pos_2d is not None:
            self._setup_2d_skeleton(pos_2d, w2c, is_pinhole, skeleton_type)
            self._setup_camera(w2c)
        if pred_pos is not None:
            self._setup_pred_3d_skeleton(pred_pos, skeleton_type)
        if gt_pos is not None:
            self._setup_gt_3d_skeleton(gt_pos, skeleton_type)
        if c_pos is not None:
            self._setup_cam_3d_skeleton(c_pos, w2c, skeleton_type)

        self.vis.register_animation_callback(self.animation_callback)
        self.vis.run()
        self.vis.destroy_window()

    def _setup_3d_skeleton(self, pos, upaxis, skeleton_type):
        if isinstance(pos, torch.Tensor):
            pos = torch.clone(pos).cpu().detach().numpy()
        J = pos.shape[-2]
        self.J = J
        pos = pos.reshape(pos.shape[0], -1, J, 3)  # N, T, J, 3
        self.skeleton = add_skeleton(pos[0, 0], skeleton_type=skeleton_type)  # Add 3d skeleton mesh

        self.vis.add_geometry(get_ground_mesh(pos, upaxis=upaxis))

        for j in range(self.J):
            self.vis.add_geometry(self.skeleton[0][j])
        self.vis.add_geometry(self.skeleton[1])
        self.pos = pos

    def _setup_2d_skeleton(self, pos_2d, w2c, is_pinhole, skeleton_type):
        if isinstance(pos_2d, torch.Tensor):
            pos_2d = torch.clone(pos_2d).cpu().detach().numpy()
        if isinstance(w2c, torch.Tensor):
            w2c = torch.clone(w2c).cpu().detach().numpy()
        pos_2d = pos_2d.reshape(pos_2d.shape[:2] + (-1, self.J, 2))  # N, V, T, J, 2
        self.cam_skeleton = []  # skeleton on camera plane, z = 0.25
        self.cam_skeleton_line = []  # line start from skeleton to z = 1.5 on camera plane
        self.cam_skeleton_pos = []  # start of line
        self.cam_skeleton_pos_end = []  # end of line
        # Add 2d skeleton mesh of each view
        for i in range(pos_2d.shape[1]):
            max_z, min_z = get_good_z_for_2dvis(w2c[:, :, i], self.pos[..., 0, :], is_pinhole)
            pos_3d_ = pos_2dto3d(pos_2d[:, i], w2c[:, :, i], z=min_z, is_pinhole=is_pinhole)  # N, T, J, 3
            self.cam_skeleton.append(add_skeleton(pos_3d_[0, 0], r=0.025, skeleton_type=skeleton_type))
            self.cam_skeleton_pos.append(pos_3d_[:, None])
            for j in range(self.J):
                self.vis.add_geometry(self.cam_skeleton[i][0][j])
            self.vis.add_geometry(self.cam_skeleton[i][1])

            pos_3d_e = pos_2dto3d(pos_2d[:, i], w2c[:, :, i], z=max_z, is_pinhole=is_pinhole)
            self.cam_skeleton_line.append(add_line(pos_3d_[0, 0], pos_3d_e[0, 0]))
            self.vis.add_geometry(self.cam_skeleton_line[i])
            self.cam_skeleton_pos_end.append(pos_3d_e[:, None])
        self.cam_skeleton_pos = np.concatenate(self.cam_skeleton_pos, axis=1)
        self.cam_skeleton_pos_end = np.concatenate(self.cam_skeleton_pos_end, axis=1)
        self.pos_2d = pos_2d
        self.w2c = w2c

    def _setup_camera(self, w2c):
        if isinstance(w2c, torch.Tensor):
            w2c = torch.clone(w2c).cpu().detach().numpy()
        cam_coords = []
        cam_meshs = []
        for i, w2c_ in enumerate(w2c[0, 0]):
            cam_lineset = add_camera_line_set(w2c_)
            self.vis.add_geometry(cam_lineset)
            cam_coords.append(cam_lineset)
            size = 0.5 if i == 0 else 0.2
            cam_mesh = get_coordinate_mesh(mat=np.linalg.inv(w2c_), size=size)
            self.vis.add_geometry(cam_mesh)
            cam_meshs.append(cam_mesh)
        self.cam_coord = cam_coords
        self.cam_mesh = cam_meshs
        self.prev_w2c = w2c[0, 0]

    def _setup_pred_3d_skeleton(self, pos, skeleton_type):
        if isinstance(pos, torch.Tensor):
            pos = torch.clone(pos).cpu().detach().numpy()
        pos = pos.reshape(pos.shape[:2] + (-1, self.J, 3))  # N, V, T, J, 3
        self.pred_skeleton = []
        for i in range(pos.shape[1]):
            self.pred_skeleton.append(add_skeleton(pos[0, i, 0], r=0.025, skeleton_type=skeleton_type))
            for j in range(self.J):
                self.vis.add_geometry(self.pred_skeleton[i][0][j])
            self.vis.add_geometry(self.pred_skeleton[i][1])
        self.pred_pos = pos

    def _setup_gt_3d_skeleton(self, pos, skeleton_type):
        if isinstance(pos, torch.Tensor):
            pos = torch.clone(pos).cpu().detach().numpy()
        gt_J = pos.shape[-2]
        self.gt_J = gt_J
        pos = pos.reshape(pos.shape[:2] + (-1, self.gt_J, 3))  # N, V, T, J, 3
        self.gt_skeleton = []
        for i in range(pos.shape[1]):
            self.gt_skeleton.append(add_skeleton(pos[0, i, 0], r=0.04, skeleton_type=skeleton_type))
            for j in range(self.gt_J):
                self.vis.add_geometry(self.gt_skeleton[i][0][j])
            self.vis.add_geometry(self.gt_skeleton[i][1])
        self.gt_pos = pos

    def _setup_cam_3d_skeleton(self, c_pos, w2c, skeleton_type):
        if isinstance(c_pos, torch.Tensor):
            c_pos = torch.clone(c_pos).cpu().detach().numpy()
        if isinstance(w2c, torch.Tensor):
            w2c = torch.clone(w2c).cpu().detach().numpy()
        c_pos = c_pos.reshape(c_pos.shape[:2] + (-1, self.J, 3))  # N, V, T, J, 2
        self.cam_3d_skeleton = []  # skeleton in camera view
        # Add 3d skeleton mesh of each camera view
        for i in range(c_pos.shape[1]):
            c2w = np.linalg.inv(w2c[:, :, i])  # N, T, 4, 4
            c_pos_shape = c_pos[:, i].shape  # N, T, J, 3
            c_pos_ = c_pos[:, i].reshape(-1, c_pos_shape[-2], 3)  # (N*T), J, 3
            c_pos_ = matrix.get_position_from(c_pos_, c2w.reshape(-1, 4, 4))
            c_pos_ = c_pos_.reshape(c_pos_shape[:-1] + (3,))
            self.cam_3d_skeleton.append(add_skeleton(c_pos_[0, 0], r=0.03, skeleton_type=skeleton_type))
            for j in range(self.J):
                self.vis.add_geometry(self.cam_3d_skeleton[i][0][j])
            self.vis.add_geometry(self.cam_3d_skeleton[i][1])

        self.c_pos = c_pos

    def animation_callback(self, vis):
        if STATE["play"]:
            self.time_i += 1
            if self.time_i >= self.pos.shape[1]:
                self.time_i = 0
        if STATE["next"]:
            self.time_i += 1
            STATE["next"] = False
        if STATE["back"]:
            self.time_i -= 1
            STATE["back"] = False
        if STATE["reset"]:
            self.time_i = 0
            STATE["reset"] = False
        if STATE["after"]:
            self.iter_i += 1
            STATE["after"] = False
        if STATE["prev"]:
            self.iter_i -= 1
            STATE["prev"] = False
        print(f"STEP: {self.iter_i}, TIME: {self.time_i}. Press 'H' for help guide.")
        self.time_i = min(self.pos.shape[1] - 1, self.time_i)
        self.time_i = max(0, self.time_i)
        self.iter_i = min(self.pos.shape[0] - 1, self.iter_i)
        self.iter_i = max(0, self.iter_i)

        if VIS_STATE[1]:
            self.update_skeleton(self.pos[self.iter_i, self.time_i], self.skeleton)
        else:
            self.disable_skeleton(self.pos[self.iter_i, self.time_i] * 100, self.skeleton)
        if self.pos_2d is not None:
            for i in range(self.cam_skeleton_pos.shape[1]):
                if VIS_STATE[2]:
                    self.update_skeleton(self.cam_skeleton_pos[self.iter_i, i, self.time_i], self.cam_skeleton[i])
                    self.update_line(
                        self.cam_skeleton_pos[self.iter_i, i, self.time_i],
                        self.cam_skeleton_pos_end[self.iter_i, i, self.time_i],
                        self.cam_skeleton_line[i],
                    )
                else:
                    self.disable_skeleton(self.cam_skeleton_pos[self.iter_i, i, self.time_i], self.cam_skeleton[i])
                    self.disable_line(
                        self.cam_skeleton_pos[self.iter_i, i, self.time_i],
                        self.cam_skeleton_pos_end[self.iter_i, i, self.time_i],
                        self.cam_skeleton_line[i],
                    )
            self.update_camera(self.w2c[self.iter_i, self.time_i])

        if self.pred_pos is not None:
            for i in range(self.pred_pos.shape[1]):
                if VIS_STATE[3]:
                    self.update_skeleton(self.pred_pos[self.iter_i, i, self.time_i], self.pred_skeleton[i])
                else:
                    self.disable_skeleton(self.pred_pos[self.iter_i, i, self.time_i], self.pred_skeleton[i])
        if self.gt_pos is not None:
            for i in range(self.gt_pos.shape[1]):
                if VIS_STATE[4]:
                    self.update_skeleton(self.gt_pos[self.iter_i, i, self.time_i], self.gt_skeleton[i])
                else:
                    self.disable_skeleton(self.gt_pos[self.iter_i, i, self.time_i], self.gt_skeleton[i])
        if self.c_pos is not None:
            for i in range(self.c_pos.shape[1]):
                if VIS_STATE[5]:
                    self.update_skeleton(self.c_pos[self.iter_i, i, self.time_i], self.cam_3d_skeleton[i])
                else:
                    self.disable_skeleton(self.c_pos[self.iter_i, i, self.time_i], self.cam_3d_skeleton[i])

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_skeleton(self, pos, skeleton):
        for i in range(pos.shape[-2]):
            skeleton[0][i].translate(pos[i], relative=False)
            self.vis.update_geometry(skeleton[0][i])
        skeleton[1].points = o3d.utility.Vector3dVector(pos)
        self.vis.update_geometry(skeleton[1])

    def disable_skeleton(self, pos, skeleton):
        # Move it to very far places to disable it.
        for i in range(pos.shape[-2]):
            skeleton[0][i].translate(np.ones_like(pos[i]) * 100 + 100, relative=False)
            self.vis.update_geometry(skeleton[0][i])
        skeleton[1].points = o3d.utility.Vector3dVector(np.ones_like(pos) * 100 + 100)
        self.vis.update_geometry(skeleton[1])

    def update_line(self, pos_s, pos_e, line):
        pos = np.concatenate((pos_s, pos_e), axis=0)
        line.points = o3d.utility.Vector3dVector(pos)
        self.vis.update_geometry(line)

    def disable_line(self, pos_s, pos_e, line):
        # Move it to very far places to disable it.
        pos = np.concatenate((pos_s, pos_e), axis=0)
        line.points = o3d.utility.Vector3dVector(np.ones_like(pos) * 100 + 100)
        self.vis.update_geometry(line)

    def update_camera(self, w2c):
        points = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        for i, w2c_ in enumerate(w2c):
            mat = np.linalg.inv(w2c_)
            prev_mat_inv = self.prev_w2c[i]
            cam_line_pts = matrix.get_position_from(np.array(points), mat)
            self.cam_coord[i].points = o3d.utility.Vector3dVector(cam_line_pts)
            self.vis.update_geometry(self.cam_coord[i])
            self.cam_mesh[i].transform(prev_mat_inv)
            self.cam_mesh[i].transform(mat)
            self.vis.update_geometry(self.cam_mesh[i])
        self.prev_w2c = w2c


def vis_smpl_forward_animation(transl, pose):
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
    ]
    colors = [COLORS[0], COLORS[2], COLORS[2], COLORS[1], COLORS[2]]
    ground_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_line_set.colors = o3d.utility.Vector3dVector(colors)
    smpl_model = smplx.create(
        "./inputs/checkpoints/body_models",
        model_type="smplh",
        gender="male",
        num_betas=16,
        batch_size=1,
    )

    output = smpl_model(
        body_pose=pose[:1, 3:],
        global_orient=pose[:1, :3],
        transl=transl[:1],
        return_verts=True,
    )
    verts = output.vertices.detach().cpu().numpy()

    def play_callback(vis, key, action):
        print("play", "key", key, "action", action)
        STATE["play"] = True

    def stop_callback(vis, key, action):
        print("stop", "key", key, "action", action)
        STATE["play"] = False

    def reset_callback(vis, key, action):
        print("reset", "key", key, "action", action)
        STATE["reset"] = True
        STATE["play"] = False

    def next_callback(vis, key, action):
        print("next", "key", key, "action", action)
        STATE["next"] = True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(ord("P"), play_callback)
    vis.register_key_action_callback(ord("S"), stop_callback)
    vis.register_key_action_callback(ord("R"), reset_callback)
    vis.register_key_action_callback(ord("N"), next_callback)

    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
    smpl_mesh.compute_vertex_normals()
    smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])

    vis.add_geometry(smpl_mesh)
    vis.add_geometry(ground_line_set)

    MAX_N = pose.shape[0]
    i = 0

    def animation_callback(vis):
        nonlocal i
        if STATE["play"]:
            i += 1
        if STATE["next"]:
            i += 1
            STATE["next"] = False
        if i >= MAX_N:
            i = MAX_N - 1
        if STATE["reset"]:
            i = 0
            STATE["reset"] = False
        output = smpl_model(
            body_pose=pose[i : i + 1, 3:],
            global_orient=pose[i : i + 1, :3],
            transl=transl[i : i + 1],
            return_verts=True,
        )
        verts = output.vertices.detach().cpu().numpy()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
        smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])
        vis.update_geometry(smpl_mesh)
        vis.poll_events()
        vis.update_renderer()

    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()


def vis_smpl_verts_animation(verts, faces):
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
    ]
    colors = [COLORS[0], COLORS[2], COLORS[2], COLORS[1], COLORS[2]]
    ground_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_line_set.colors = o3d.utility.Vector3dVector(colors)

    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()

    def play_callback(vis, key, action):
        print("play", "key", key, "action", action)
        STATE["play"] = True

    def stop_callback(vis, key, action):
        print("stop", "key", key, "action", action)
        STATE["play"] = False

    def reset_callback(vis, key, action):
        print("reset", "key", key, "action", action)
        STATE["reset"] = True
        STATE["play"] = False

    def next_callback(vis, key, action):
        print("next", "key", key, "action", action)
        STATE["next"] = True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(ord("P"), play_callback)
    vis.register_key_action_callback(ord("S"), stop_callback)
    vis.register_key_action_callback(ord("R"), reset_callback)
    vis.register_key_action_callback(ord("N"), next_callback)

    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
    smpl_mesh.compute_vertex_normals()
    smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])

    vis.add_geometry(smpl_mesh)
    vis.add_geometry(ground_line_set)

    MAX_N = verts.shape[0]
    i = 0

    def animation_callback(vis):
        nonlocal i
        if STATE["play"]:
            i += 1
        if STATE["next"]:
            i += 1
            STATE["next"] = False
        if i >= MAX_N:
            i = MAX_N - 1
        if STATE["reset"]:
            i = 0
            STATE["reset"] = False
        print(f"{i}/{MAX_N}")
        smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[i])
        smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])
        vis.update_geometry(smpl_mesh)
        vis.poll_events()
        vis.update_renderer()

    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()
