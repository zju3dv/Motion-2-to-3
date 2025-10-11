import os
import time
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.wis3d_utils import convert_motion_as_line_mesh, draw_T_w2c, make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background, simple_render_mesh
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.phc.pos2smpl import convert_pos_to_smpl
from hmr4d.utils.video_io_utils import save_video
import pyrender
import trimesh
from transformers import CLIPProcessor, CLIPModel

import numpy as np
from hmr4d.utils.vis.pyrender_utils import get_campose_pyrender
from tqdm import tqdm


width, height = 512, 512
W, H = 1024, 1024

os.environ["PYOPENGL_PLATFORM"] = "egl"
model = CLIPModel.from_pretrained("inputs/checkpoints/huggingface/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("inputs/checkpoints/huggingface/clip-vit-base-patch32")

smplh_model = make_smplx("rich-smplh", gender="neutral").cuda()

# USE_SMPL = True
USE_SMPL = False
USE_PYTORCH3D = True
# MODEL = "ours"
MODEL = "mdm"
path = f"./outputs/dumped_newtext_{MODEL}"
print(f"Load from {path}")
saved_motions = os.listdir(path)
saved_motions.sort()
saved_motions = [os.path.join(path, p) for p in saved_motions if "pth" in p]
p_i = 0
all_clip_score = []
for p in tqdm(saved_motions):
    motion = torch.load(p, map_location="cpu")
    text = motion["text"]
    length = motion["length"]
    joints = motion["pred"]
    joints = joints[:length]

    if USE_SMPL:
        joints = joints.cuda()
        smpl_pose = convert_pos_to_smpl(joints)
        transl = smpl_pose[:, :3]
        global_orient = smpl_pose[:, 3:6]
        body_pose = smpl_pose[:, 6:]
        smplh_model_output = smplh_model(transl=transl, global_orient=global_orient, body_pose=body_pose)
        vertices = smplh_model_output.vertices
        vertices = vertices.cpu()
        fit_joints = smplh_model_output.joints
        fit_joints = fit_joints.cpu()
        faces = smplh_model.bm.faces
        vertices = vertices - fit_joints[:, [0], :]
    else:
        j3d_0offset = joints - joints[:, [0], :]
        vertices, faces, vertex_colors = convert_motion_as_line_mesh(j3d_0offset)

    if USE_PYTORCH3D: 
        K = create_camera_sensor(W, H, 24)[2]
        focal_length = K[0, 0]  # 24mm lens

        verts_render = vertices + torch.tensor([0.0, 0.0, 2.0])
        verts_render = verts_render * torch.tensor([1, -1, 1])
        render_dict = {
            "whf": (W, H, focal_length),
            "faces": faces,
            "verts": verts_render,
        }
        images = simple_render_mesh(render_dict, VI=1)

        rendered_imgs = [cv2.resize(img, dsize=np.array([512, 512])) for img in images]
    else:
        ### pyrender ####
        rendered_imgs = []
        for i in tqdm(range(vertices.shape[0])):
            scene = pyrender.Scene()

            # Add Person
            example_trimesh = trimesh.Trimesh(vertices=vertices[i], faces=faces)
            mesh = pyrender.Mesh.from_trimesh(example_trimesh)
            scene.add(mesh)

            # Add Camera
            cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
            p_from = np.array([0.0, 0.2, 2.0])
            p_to = np.array([0.0, 0.0, 0.0])
            gravity_vec = np.array([0.0, -1.0, 0.0])
            cam_pose = get_campose_pyrender(p_from, p_to, gravity_vec)
            scene.add(cam, pose=cam_pose)

            # Add Light
            light = pyrender.SpotLight(
                color=np.array([0.8, 0.8, 0.8]), intensity=5.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
            )
            scene.add(light, pose=cam_pose)
            color, depth = pyrender.OffscreenRenderer(width, height).render(scene)
            # Image.fromarray(color)
            rendered_imgs.append(color)
        ################

    os.makedirs(f"./visualizations/render/{MODEL}", exist_ok=True) 
    save_video(rendered_imgs, f"./visualizations/render/{MODEL}/{p_i}.mp4")
    rendered_imgs = np.array(rendered_imgs)


    with torch.no_grad():
        # process all frames
        inputs = processor(text=[text], images=rendered_imgs, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)

        # mean over all frames
        logits_per_image = (outputs.logits_per_image).mean()  # this is the image-text similarity score
        print(f"{p_i}: {logits_per_image.item():.2f} - {text}\n")
        all_clip_score.append(logits_per_image.item())

    p_i += 1

    print(f"{MODEL}: {p_i} samples, avg clip score: {np.mean(all_clip_score):.2f}")