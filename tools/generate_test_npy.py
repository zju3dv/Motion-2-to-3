import os
import cv2
import glob
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo_transform import apply_T_on_points, project_p2d
from hmr4d.dataset.rich.rich_utils import get_cam2params

def parse_args():
    """ Parse all the args. """
    parser = argparse.ArgumentParser(description="Be used to generagte test numpy file.")
    parser.add_argument("--video", type=str, default=None, help="path to single video sequence folder")
    parser.add_argument("--gender", type=str, default=None, help="select the gender of the picture")
    parser.add_argument("--suffix", type=str, default="jpeg", help="suffix of frame image")
    parser.add_argument("--interval", type=int, default=1, help="interval between frames to be sampled")
    parser.add_argument("--limit", type=int, default=300, help="number of frames to be sampled")
    parser.add_argument("--output_dir", type=str, default=None, help="path to output numpy file")
    parser.add_argument("--data_root", type=str, default="inputs/RICH", help="path to scene info root")
    parser.add_argument("--start_frame", type=int, default=0, help="index of the first frame selected")
    parser.add_argument("--scan_name", type=str, default="scan_camcoord", help="some scene might have more than one scan")
    args = parser.parse_args()
    
    check_null_list = ["video", "output_dir", "gender"]
    for check in check_null_list:
        if getattr(args, check) is None:
            print(f"arguments --{check} can't be None!")
        
    return args

def load_smpl_verts_w(args):
    """ Get smpl verts @ world coordinates. """
    # get smplh model   
    smplh = make_smplx("rich-smplh", gender=args.gender)
    # load smplh params
    gt_path = os.path.join(f"{args.data_root}/test_body/{scene_name}_{sbj_id}_{task_name}/{frame_id}/{sbj_id}_smplh.pkl")
    with open(gt_path, "rb") as f:
        data = pickle.load(f)
    smplh_params = {k: torch.from_numpy(v).float().reshape(1, -1) for k, v in data.items()}
    # get smplh verts
    smplh_opt = smplh(**smplh_params)
    verts_3d_w = smplh_opt.vertices
    return verts_3d_w

def compute_tight_bbx(verts_3d_w, T_w2c, K):
    """ Get the bbx by project the samplh model to the camera. """
    # move to cuda
    T_w2c, K = T_w2c.cuda(), K.cuda()
    verts_3d_w = verts_3d_w.cuda()
    # project to 2d
    verts_3d_c = apply_T_on_points(verts_3d_w, T_w2c[None])
    verts_2d = project_p2d(verts_3d_c, K[None])[0]
    # get the tight bbx
    min_2d = verts_2d.T.min(-1)[0]
    max_2d = verts_2d.T.max(-1)[0]
    bbx = torch.stack([min_2d, max_2d]).reshape(-1).cpu().numpy()
    return bbx

def compute_square_bbx(lurb_tight, margin:float=51.5*4):
    """ Get the square bbx with a margin. """
    side_lens = [ lurb_tight[2] - lurb_tight[0], lurb_tight[3] - lurb_tight[1] ]
    square_side = np.max(side_lens) + margin
    center = [ (lurb_tight[2] + lurb_tight[0]) / 2, (lurb_tight[3] + lurb_tight[1]) / 2 ]
    lurb_square = np.array(
        [
            center[0] - square_side / 2, 
            center[1] - square_side / 2, 
            center[0] + square_side / 2, 
            center[1] + square_side / 2
        ], 
        dtype=np.float32
    )
    return lurb_square

def visualize_bbx(img_path, lurb_tight, lurb_square, scale=4):
    """ Visualize the bbx by print the rectangle on the picture. """
    img = cv2.imread(img_path)
    lurb_tight_resized = lurb_tight / scale
    lurb_square_resized = lurb_square / scale
    cv2.rectangle(
        img, 
        ( int(lurb_square_resized[0]), int(lurb_square_resized[1]) ), 
        ( int(lurb_square_resized[2]), int(lurb_square_resized[3]) ), 
        (0, 255, 0), 
        2
    )
    cv2.rectangle(
        img, 
        ( int(lurb_tight_resized[0]), int(lurb_tight_resized[1]) ), 
        ( int(lurb_tight_resized[2]), int(lurb_tight_resized[3]) ), 
        (0, 0, 255), 
        2
    )
    cv2.imwrite("vis.jpeg", img)

def cut_img_with_bbx(img_path, out_path, lurb, scale=4):
    """ Cut the image with the given params. """
    img = cv2.imread(img_path)
    # scale the lurb as the picture is downsampled by 4
    lurb_resized = lurb / scale
    img = img[
        int(lurb_resized[1]) : int(lurb_resized[3]),
        int(lurb_resized[0]) : int(lurb_resized[2]),
    ]
    # resize the output picture size to 224 * 224
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(out_path, img)

def get_bbx_and_verts(args, meta):
    """ Get the bbx of the image and also the side len of square. """
    # 1. get the camera parameters 
    scene_info_path = os.path.join(args.data_root, "sahmr_support", "scene_info")
    T_w2c, K = get_cam2params(scene_info_path)[meta["cam_key"]]

    # 2. calculate the bbx lurb
    verts_w = load_smpl_verts_w(args)   # (1, 6890, 3)
    lurb_tight = compute_tight_bbx(verts_w, T_w2c, K)
    lurb_square = compute_square_bbx(lurb_tight)
    
    # 3. get and draw frame on the image
    # visualize_bbx(meta["frame_path"], lurb_tight, lurb_square)
    
    return lurb_square, verts_w[0].cpu().numpy()

def update_bbx(lurb, max_len):
    """ We should keep all the lurb to cut is at the same size. """
    local_len = lurb[2] - lurb[0]
    delta_len = max_len - local_len
    half_delta = delta_len / 2
    lurb[:2] -= half_delta
    lurb[-2:] += half_delta
    return lurb

if __name__ == "__main__":
    # some switches
    cut_img, dump_body = True, True

    # get args
    args = parse_args()
    
    # 1. get the frame image paths to be processed
    frame_root = os.path.join(args.video)
    frame_paths = sorted(glob.glob(os.path.join(frame_root, f"*.{args.suffix}")))
    frame_paths = frame_paths[args.start_frame::args.interval][:args.limit]
    
    # 2. processe all meta data the output file needs
    metas = []
    for frame_path in tqdm(frame_paths):
        # 2.1. get raw data from frame path
        # `frame_path` should be like f".../test/{scene_name}_{sbj_id}_{task_name}/cam_{cam_id}/{frame_id}_{cam_id}.jpeg" 
        parts = frame_path.split("/")
        # all these will be represented as str here, which means leading zero exists
        frame_id = parts[-1].split("_")[0]
        cam_id = parts[-2].split("_")[-1]
        scene_name, sbj_id, task_name = parts[-3].split("_")
        raw = {
            "frame_path": frame_path,
            "frame_id"  : frame_id,
            "cam_id"    : cam_id,
            "scene_name": scene_name,
            "sbj_id"    : sbj_id,
            "task_name" : task_name
        }
        # print(f"frame id: {frame_id}, cam id: {cam_id}, scene name: {scene_name}, sbj id: {sbj_id}, task name: {task_name}")

        # 2.2. combined the raw data to get the needed information
        meta = {}
        # get simple meta data
        meta["raw"] = raw
        
        issue_name = f"{scene_name}_{sbj_id}_{task_name}_{int(cam_id)}"

        meta["frame_path"] = frame_path
        # key info
        meta["gender"]     = args.gender
        meta["cap_name"]   = scene_name
        meta["scan_name"]  = args.scan_name
        meta["frame_name"] = frame_id
        meta["issue_name"] = issue_name
        meta["img_key"]    = f"{scene_name}_{sbj_id}_{task_name}_{int(cam_id)}_{frame_id}_{sbj_id}"
        meta["cam_key"]    = f"{scene_name}_{int(cam_id)}"
        meta["scene_key"]  = f"{scene_name}_{args.scan_name}"
        # support files
        meta["contact_file"] = f"contact/{scene_name}_{sbj_id}_{task_name}/{frame_id}/{sbj_id}.npy"
        meta["img_file"]     = f"{issue_name}_{frame_id}"
        meta["body_file"]    = f"{issue_name}_{frame_id}"

        # get the bbx and verts
        square_lurb, verts = get_bbx_and_verts(args, meta)

        # prepare complex meta data
        meta["test_squared_bbx_lurb"] = square_lurb
        
        # 2.3. generate cutted image according to the bbx
        img_dir = f"cutted_imgs/{scene_name}_{sbj_id}_{task_name}_{int(cam_id)}"
        meta["img_file"] = f"{img_dir}/{meta['img_file']}.png"
        if cut_img:
            img_path = f"{args.output_dir}/{meta['img_file']}"
            os.makedirs(f"{args.output_dir}/{img_dir}", exist_ok=True)
            cut_img_with_bbx(meta["frame_path"], img_path, square_lurb)

        # 2.4. generate verts.npy
        body_dir = f"body_verts/{scene_name}_{sbj_id}_{task_name}_{int(cam_id)}"
        meta['body_file'] = f"{body_dir}/{meta['body_file']}.npy"
        if dump_body:
            body_path = f"{args.output_dir}/{meta['body_file']}"
            os.makedirs(f"{args.output_dir}/{body_dir}", exist_ok=True)
            np.save(body_path, verts)

        metas.append(meta)
    
    # 3. store the metas
    metas = np.array(metas)
    
    meta_dir = f"{args.output_dir}/meta"
    os.makedirs(meta_dir, exist_ok=True)
    np.save(f"{meta_dir}/{issue_name}.npy", metas)