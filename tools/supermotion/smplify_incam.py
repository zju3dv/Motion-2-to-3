import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import imageio
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo_transform import apply_T_on_points, project_p2d
from hmr4d.utils.vis.vis_kpts import draw_conf_kpts_cv2
import decord
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer
from hmr4d.utils.geo.hmr_cam import compute_transl_full_cam, perspective_projection
from hmr4d.model.smplify.losses import SMPLifyLoss
from hmr4d.utils.eval.eval_utils import compute_camcoord_metrics, compute_global_metrics
from hmr4d.model.supermotion.callbacks.metric_mocap import as_np_array
from hmr4d.utils.pylogger import Log

from einops import einsum


decord.bridge.set_bridge("torch")

out_folder = Path("outputs/smplify_render")
smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
J_regressor = torch.load("hmr4d/utils/body_model/smpl_coco17_J_regressor.pt").cuda()
J_regressor_eval = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
faces = make_smplx("smpl").faces

smplx_model = {
    "male": make_smplx("rich-smplx", gender="male").cuda(),
    "female": make_smplx("rich-smplx", gender="female").cuda(),
    "neutral": make_smplx("rich-smplx", gender="neutral").cuda(),
}


def render_kp2d_overlay(images, kp2d_render, conf=None):
    if conf is None:  # Set to full confidence
        conf = torch.ones_like(kp2d_render[..., 0])

    kp2d_overlays = []
    for i, img in enumerate(images):
        img_kp2d_overlay = draw_conf_kpts_cv2(img, kp2d_render[i], conf[i], thickness=2)
        kp2d_overlays.append(img_kp2d_overlay)

    # use imageio to save video
    video_path = str(out_folder / "kp2d_overlay.mp4")
    imageio.mimwrite(video_path, kp2d_overlays, fps=30)


def render_smpl_overlay(vertices_render, images, K_render, seq_length=None, vname="debug"):
    width, height = images.shape[2], images.shape[1]
    renderer = Renderer(width, height, device="cuda", faces=faces, K=K_render)
    out_fn = str(out_folder / f"render_{vname}.mp4")
    writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)
    if seq_length is None:
        seq_length = len(vertices_render)
    for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
        img_overlay = renderer.render_mesh(vertices_render[i].cuda(), images[i])
        writer.append_data(img_overlay)
    writer.close()


def get_render_images_K(dump):
    # imgs
    video_path = dump["meta_render"]["video_path"]
    frame_id = dump["meta_render"]["frame_id"].cpu().numpy()
    vr = decord.VideoReader(video_path)
    images = vr.get_batch(list(frame_id)).numpy()  # (F, H/4, W/4, 3), uint8, numpy
    # corresponding K
    K_render = dump["meta_render"]["K"]

    return images, K_render


def visualize(dump):
    images, K_render = get_render_images_K(dump)
    kp2d = dump["kp2d"]  # (L, 17, 3)
    K_fullimg = dump["K_fullimg"]  # (3, 3)
    seq_length = kp2d.shape[0]

    # Render ViT-COCO17
    kp2d_render = kp2d[:, :, :2] / 4  # (L ,17, 2)
    conf = kp2d[:, :, 2]  # (L, 17)
    render_kp2d_overlay(images, kp2d_render, conf)

    # Render SMPL overlay
    smpl_params = dump["pred_smpl_params_incam"]
    smplx_model = make_smplx("supermotion")
    smplx_output = smplx_model.forward(**smpl_params)

    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_output.vertices])
    pred_c_j3d = einsum(J_regressor, pred_c_verts, "j v, l v i -> l j i")
    pred_j2d = project_p2d(pred_c_j3d, K_render)
    # render_kp2d_overlay(images, pred_j2d)
    render_smpl_overlay(pred_c_verts, images, K_render, 60, vname="pred")


def run_smplify(dump, render_result=False):
    # Prepare rendering utilities
    images, K_render = get_render_images_K(dump)

    # Helper
    kp2d = dump["kp2d"].cuda()
    K_fullimg = dump["K_fullimg"].cuda()
    bbx_xys = dump["bbx_xys"].cuda()
    helper_dict = {
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "kp2d": kp2d,
    }

    # Params
    param_dict = {
        "body_pose": dump["pred_smpl_params_incam"]["body_pose"],
        "global_orient": dump["pred_smpl_params_incam"]["global_orient"],
        "betas": dump["pred_smpl_params_incam"]["betas"],
        "transl": dump["pred_smpl_params_incam"]["transl"],
    }
    param_dict = {k: v.detach().float().cuda().requires_grad_(True) for k, v in param_dict.items()}

    lr = 1e-2
    num_iters = 5
    num_steps = 10

    # Stage 1. Optimize translation
    optimizer = torch.optim.LBFGS([param_dict["transl"]], lr=lr, max_iter=num_iters, line_search_fn="strong_wolfe")
    loss_fn = SMPLifyLoss(smplx_model["neutral"], param_dict)
    closure = loss_fn.create_closure(optimizer, param_dict, helper_dict)

    pbar = tqdm(range(num_steps), leave=False)
    for j in pbar:
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        pbar.set_description(f"Loss: {loss.item():.1f}")

    # Stage 2. Optimize all
    seq_length = kp2d.shape[0]
    param_dict = {k: v.detach().float().cuda().requires_grad_(True) for k, v in param_dict.items()}
    optimizer = torch.optim.LBFGS(
        param_dict.values(), lr=lr * seq_length, max_iter=num_iters, line_search_fn="strong_wolfe"
    )
    closure = loss_fn.create_closure(optimizer, param_dict, helper_dict)

    pbar = tqdm(range(num_steps), leave=False)
    for j in pbar:
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        pbar.set_description(f"Loss: {loss.item():.1f}")

    # Result
    with torch.no_grad():
        # Prediction
        smplx_output = smplx_model["neutral"].forward(
            body_pose=param_dict["body_pose"],
            global_orient=param_dict["global_orient"],
            betas=param_dict["betas"],
            transl=param_dict["transl"],
        )
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_output.vertices])
        pred_c_j3d = einsum(J_regressor_eval, pred_c_verts, "j v, l v i -> l j i")
        offset = pred_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
        pred_cr_j3d = pred_c_j3d - offset
        pred_cr_verts = pred_c_verts - offset

        pred = {
            "pred_c_verts": pred_c_verts,
            "pred_c_j3d": pred_c_j3d,
            "pred_cr_verts": pred_cr_verts,
            "pred_cr_j3d": pred_cr_j3d,
        }

    # Visualize result
    if render_result:
        name = dump["meta_render"]["name"]
        render_smpl_overlay(pred_c_verts, images, K_render, vname=f"smplify-{name}")

    return pred


def get_init_pred(dump):
    smpl_param_incam = {
        "body_pose": dump["pred_smpl_params_incam"]["body_pose"],
        "global_orient": dump["pred_smpl_params_incam"]["global_orient"],
        "betas": dump["pred_smpl_params_incam"]["betas"],
        "transl": dump["pred_smpl_params_incam"]["transl"],
    }
    smpl_param_incam = {k: v.cuda() for k, v in smpl_param_incam.items()}
    smplx_out = smplx_model["neutral"](**smpl_param_incam)
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    pred_c_j3d = torch.matmul(J_regressor_eval, pred_c_verts)
    offset = pred_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
    pred_cr_j3d = pred_c_j3d - offset
    pred_cr_verts = pred_c_verts - offset
    pred = {
        "pred_c_verts": pred_c_verts,
        "pred_c_j3d": pred_c_j3d,
        "pred_cr_verts": pred_cr_verts,
        "pred_cr_j3d": pred_cr_j3d,
    }
    pred = {k: v.detach().cpu() for k, v in pred.items()}
    return pred


def get_groundtruth(dump):
    # Groundtruth
    gender = dump["gender"]
    T_w2c = dump["T_w2c"].cuda()
    target_w_params = {k: v.cuda() for k, v in dump["gt_smpl_params_world"].items()}
    target_w_output = smplx_model[gender](**target_w_params)
    target_w_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in target_w_output.vertices])
    target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
    target_c_j3d = torch.matmul(J_regressor_eval, target_c_verts)
    offset = target_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
    target_cr_j3d = target_c_j3d - offset
    target_cr_verts = target_c_verts - offset

    target = {
        "target_c_verts": target_c_verts,
        "target_c_j3d": target_c_j3d,
        "target_cr_verts": target_cr_verts,
        "target_cr_j3d": target_cr_j3d,
    }
    target = {k: v.detach().cpu() for k, v in target.items()}
    return target


def compute_metrics(pred, target):
    batch_eval = {
        "pred_j3d": pred["pred_cr_j3d"],
        "pred_verts": pred["pred_cr_verts"],
        "target_j3d": target["target_cr_j3d"],
        "target_verts": target["target_cr_verts"],
    }
    return compute_camcoord_metrics(batch_eval)


if __name__ == "__main__":
    out_folder.mkdir(exist_ok=True, parents=True)
    dumps = torch.load("outputs/dump/ditv1-extra_pred_cam-loss2.pt")

    # visualize(dumps[0])
    # run_smplify(dumps[0])
    # run_smplify(dumps[5])

    # Prepare
    metric_aggregator_init = {
        "pa_mpjpe": {},
        "mpjpe": {},
        "pve": {},
        "accel": {},
    }
    metric_aggregator_smplify = {
        "pa_mpjpe": {},
        "mpjpe": {},
        "pve": {},
        "accel": {},
    }
    for i in tqdm(range(len(dumps))):
        vid = dumps[i]["meta_render"]["name"]

        pred = get_init_pred(dumps[i])
        groundtruth = get_groundtruth(dumps[i])
        metrics_init = compute_metrics(pred, groundtruth)
        for k in metrics_init:
            metric_aggregator_init[k][vid] = as_np_array(metrics_init[k])

        # smplify
        pred_smplify = run_smplify(dumps[i], render_result=False)
        metrics_smplify = compute_metrics(pred_smplify, groundtruth)
        for k in metrics_smplify:
            metric_aggregator_smplify[k][vid] = as_np_array(metrics_smplify[k])

    print("-----------------------------")
    # average over all batches
    metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in metric_aggregator_init.items()}
    Log.info("Initial metrics:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")
    metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in metric_aggregator_smplify.items()}
    Log.info("Smplify metrics:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")

    # print for each sequence: seqname, metric1, metric2, metric3...
    for metric_name in metric_aggregator_init:
        mm_per_seq_init = {k: v.mean() for k, v in metric_aggregator_init[metric_name].items()}
        mm_per_seq_smplify = {k: v.mean() for k, v in metric_aggregator_smplify[metric_name].items()}
        sorted_mm_per_seq_init = sorted(mm_per_seq_init.items(), key=lambda x: x[1], reverse=True)
        sorted_mm_per_seq_smplify = [(k, mm_per_seq_smplify[k]) for k, _ in sorted_mm_per_seq_init]
        Log.info(
            f"{metric_name} per sequence\n"
            + "\n".join(
                [
                    f"{v1:5.1f} -> {v2:5.1f} ({v2-v1:5.1f}) : {s1}"
                    for (s1, v1), (s2, v2) in zip(sorted_mm_per_seq_init, sorted_mm_per_seq_smplify)
                ]
            )
            + "\n------"
        )

    # In case I can't see this
    print("-----------------------------")
    # average over all batches
    metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in metric_aggregator_init.items()}
    Log.info("Initial metrics:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")
    metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in metric_aggregator_smplify.items()}
    Log.info("Smplify metrics:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")
