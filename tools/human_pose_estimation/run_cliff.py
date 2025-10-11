import torch
from hmr4d.network.hpe.cliff import CLIFF_Wrapper, default_ckpt_path


# Example data
B = 3
imgs = torch.randn((B, 224, 224, 3))
bboxs = torch.randn((B, 3))
WHs = torch.tensor([(1920, 1080)] * B)

# Model forward pass
wrapper = CLIFF_Wrapper(default_ckpt_path)
pred = wrapper.run_on_images(imgs, bboxs, WHs)

# Example Return
""" : B=3
vertices: torch.Size([3, 10475, 3])
joints3d: torch.Size([3, 127, 3])
joints2d: torch.Size([3, 127, 2])
pred_cam_t: torch.Size([3, 3])
pred_pose: torch.Size([3, 22, 3, 3])
pred_cam: torch.Size([3, 3])
pred_shape: torch.Size([3, 11])
pred_pose_6d: torch.Size([3, 132])
body_feat: torch.Size([3, 720])
body_feat2: torch.Size([3, 1024])
cam_intrinsics: torch.Size([3, 3, 3])
"""
for k, v in pred.items():
    print(f"{k}: {v.shape}")


if False:  # debug to check if I can use pred_pose to get vertices
    vertices_ = pred["vertices"]
    smplx_model = SMPLX("dataset/body_models/smplx/", num_betas=11, batch_size=len(vertices_)).cuda()
    from pytorch3d.transforms import matrix_to_axis_angle

    smplx_out = smplx_model(
        betas=pred["pred_shape"],
        body_pose=matrix_to_axis_angle(pred["pred_pose"][:, 1:22].contiguous()),
        global_orient=matrix_to_axis_angle(pred["pred_pose"][:, :1].contiguous()),
    )
    from hutil.wis3d_utils import make_wis3d

    wis3d = make_wis3d(name="debuggg")
    wis3d.add_mesh(vertices=smplx_out.vertices[0].cpu(), faces=smplx_model.faces, name="smplx_forward")
    wis3d.add_mesh(vertices=vertices_[0].cpu(), faces=smplx_model.faces, name="model_output")
