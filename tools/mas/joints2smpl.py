import torch
from tqdm import tqdm
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.phc.pos2smpl import convert_pos_to_smpl


hml3d_joints = torch.load("./inputs/hml3d/joints3d.pth")

all_keys = list(hml3d_joints.keys())
secletd_keys = all_keys[:10]

smplh_model = make_smplx("rich-smplh", gender="neutral").cuda()

for k in secletd_keys:
    joints3d = hml3d_joints[k]["joints3d"]
    joints3d = torch.tensor(joints3d, dtype=torch.float32)
    joints3d = joints3d.cuda()
    smpl_pose = convert_pos_to_smpl(joints3d)
    transl = smpl_pose[:, :3]
    global_orient = smpl_pose[:, 3:6]
    body_pose = smpl_pose[:, 6:]
    smplh_model_output = smplh_model(transl=transl, global_orient=global_orient, body_pose=body_pose)
    vertices = smplh_model_output.vertices
    fit_joints = smplh_model_output.joints
    wis3d = make_wis3d(output_dir="outputs/wis3d_debug_pos2smpl", name=f"{k}")
    add_motion_as_lines(joints3d, wis3d, name=f"input_pos")
    add_motion_as_lines(fit_joints, wis3d, name=f"fit_joints")
    for i in tqdm(range(vertices.shape[0])):
        wis3d.set_scene_id(i)
        wis3d.add_mesh(vertices[i], smplh_model.bm.faces, name="mesh")
