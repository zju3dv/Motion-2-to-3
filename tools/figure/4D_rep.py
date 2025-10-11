import torch
from tqdm import tqdm
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.camera_utils import get_camera_mat_zface, cartesian_to_spherical
# from hmr4d.utils.phc.pos2smpl import convert_pos_to_smpl
import hmr4d.utils.matrix as matrix


hml3d_joints = torch.load("./inputs/hml3d/joints3d.pth")

all_keys = list(hml3d_joints.keys())
secletd_keys = all_keys[40:41]

# smplh_model = make_smplx("rich-smplh", gender="neutral").cuda()

DISTANCE = 4.5
MAX_ANGLE = 180
ELEVA_ANGLE = 5
N_VIEWS = 4

for k in secletd_keys:
    joints_pos = hml3d_joints[k]["joints3d"]
    joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
    T_ay2ayfz = compute_T_ayfz2ay(joints_pos[:1], inverse=True)[0]  # (4, 4)
    joints_pos = apply_T_on_points(joints_pos, T_ay2ayfz)
    root_pos = joints_pos[:, :1] # (F, 1, 3)
    root_next = joints_pos[1:, :1] # (F - 1, 1, 3)
    root_next = torch.cat([root_next, root_next[-1:]], dim=0) # (F, 1, 3)
    joints_pos = torch.cat([joints_pos, root_next], dim=-2) # (F, 23, 3)
    F, J, _ = joints_pos.shape
    
    distance = torch.ones((N_VIEWS,)) * DISTANCE
    max_angle = MAX_ANGLE / 180 * torch.pi
    start = torch.rand((1,)) * 2 * torch.pi
    interval = 2 * max_angle / N_VIEWS
    angle = [start + i * interval for i in range(N_VIEWS)]
    angle = torch.cat((angle), dim=-1)
    eleva_angle = torch.ones((N_VIEWS,)) * ELEVA_ANGLE / 180.0 * torch.pi
    cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  # N, 4, 4
    cam_mat = cam_mat[None].expand(F, -1, -1, -1) # (F, N, 4, 4)
    cam_mat = matrix.set_position(cam_mat, matrix.get_position(cam_mat) + root_pos) # (F, N, 4, 4)
    T_w2c = torch.inverse(cam_mat)  # F, N, 4, 4
    c_motion = matrix.get_relative_position_to(joints_pos[:, None], cam_mat)  # F, N, J, 3
    c_motion2d = c_motion[..., :2] # just ignore z as ortho
    c_motion_in3d = torch.cat([c_motion2d, torch.ones_like(c_motion2d[..., :1])], dim=-1) # (F, N, J, 3)
    c_motion_in3d = matrix.get_position_from(c_motion_in3d, cam_mat) # (F, N, J, 3)

    if False:
        torch.save({"j3ds": joints_pos, "c_motion_in3d": c_motion_in3d, "T_w2c": T_w2c}, f"outputs/figure/pipeline_j3d.pth")
    wis3d = make_wis3d(output_dir="outputs/wis3d_4D", name=f"{k}")
    add_motion_as_lines(joints_pos[:, :-1], wis3d, name=f"joints")
    for i in range(N_VIEWS):
        add_motion_as_lines(c_motion_in3d[:, i, :-1], wis3d, name=f"view_{i}")
