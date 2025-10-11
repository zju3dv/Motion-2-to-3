import json
import hydra
import torch
import pickle
import trimesh
import numpy as np
import pytorch_lightning as pl

from time import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
from torch.utils.data import DataLoader
from hmr4d.utils.pylogger import Log, monitor_process_wrapper
from hmr4d.utils.geo_transform import apply_T_on_points
from hydra.utils import instantiate

from wis3d import Wis3D


# ================================ #
#              utils               #
# ================================ #

def L2_error(x, y):
    return (x - y).pow(2).sum(-1).sqrt()

def to_np(x):
    return x.cpu().numpy()

def to_cuda(batch, device="cuda"):
    for k in batch:
        if "meta" in k:
            continue
        if "mesh" in k or "name" in k:
            continue
        elif isinstance(batch[k], int):
            continue
        elif isinstance(batch[k], float):
            continue
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to(device) for b in batch[k]]
        elif isinstance(batch[k], dict):
            batch[k] = {_k: _v.to(device) for _k, _v in batch[k].items()}
        else:
            batch[k] = batch[k].to(device)
    return batch

def load_scene_sdfs(cfg: DictConfig):  # RICH
    # pre-load sdf
    sahmr_support_scene_dir = Path(cfg.data.opt.test.root) / "sahmr_support" / "scene_info"
    scanids = [
        "ParkingLot2/scan_camcoord",
        "LectureHall/scan_chair_scene_camcoord",
        "LectureHall/scan_yoga_scene_camcoord",
        "Gym/scan_camcoord",
        "Gym/scan_table_camcoord",
    ]

    tic = time()
    scanid_2_sdf = {}
    sdf_json_fns = [str(sahmr_support_scene_dir / f"{f}_mysdf.json") for f in scanids]
    sdf_fns = [str(sahmr_support_scene_dir / f"{f}_mysdf.npy") for f in scanids]
    for scanid, json_fn, sdf_fn in zip(scanids, sdf_json_fns, sdf_fns):
        with open(json_fn, "r") as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data["min"])).float().cuda()
            grid_max = torch.tensor(np.array(sdf_data["max"])).float().cuda()
            grid_dim = sdf_data["dim"]
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = torch.tensor(np.load(sdf_fn).reshape(grid_dim, grid_dim, grid_dim)).float().cuda()
        scanid_2_sdf[scanid] = (sdf, grid_min, grid_max, grid_dim)
    print("Pre-Load mesh and sdf:", time() - tic)
    return scanid_2_sdf


# ================================ #
#            visualize             #
# ================================ #

@monitor_process_wrapper
def get_data_loader(cfg: DictConfig) -> pl.LightningDataModule:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    return datamodule.test_dataloader()

@hydra.main(version_base=None, config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    
    vis_dir = Path(cfg.output_dir) / ".." / "vis"
    wis3d = Wis3D(vis_dir.absolute(), cfg.data.opt.test.issue_name)
    
    dump_dir = Path(cfg.output_dir) / "pred_dump"
    assert dump_dir.exists(), "Run 'tools/test_sahmr_dump.py' to dump the results first!"
    
    data_loader = get_data_loader(cfg)
    test_dir = Path(cfg.data.opt.test.root) / "sahmr_support" / "test_split"
    sdfs = load_scene_sdfs(cfg)
    
    smplx2smplh_def_pth = Path("inputs/models/model_transfer/smplx2smplh_deftrafo_setup.pkl") # TODO: should this path be in config?
    with smplx2smplh_def_pth.open("rb") as f:
        smplx2smplh_def = pickle.load(f, encoding="latin1")
    smplx2smplh_def = np.array(smplx2smplh_def["mtx"].todense(), dtype=np.float32)[:, :10475]  # (6890, 10475)
    smplx2smplh_def = torch.from_numpy(smplx2smplh_def).float().cuda()
    
    H36M_J_regressor = np.load(
        "hmr4d/network/sahmr/sahmr/data/J_regressor_h36m_correct.npy", allow_pickle=True # TODO: should this path be in config?
    )  # (17, 6890)
    H36M_J_regressor = torch.from_numpy(H36M_J_regressor).float().cuda()
    H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]
    
    for idx, batch in enumerate(tqdm(data_loader)):
        batch = to_cuda(batch)
        
        meta = batch["meta"][0]
        img_key = meta["img_key"]
        pred_path = dump_dir / f"{img_key}.pkl"
        
        if cfg.data_name == "rich_test":
            # path
            gt_path = test_dir / meta["body_file"]
            
            # scene + T_c2w
            sdf, grid_min, grid_max, grid_dim = sdfs[f"{meta['cap_name']}/{meta['scan_name']}"]
            T_c2w = batch["T_w2c"].inverse()[0]  # [4,4]
            
            # load human-scene contact mask
            hsc_mask = batch["gt_hsc"][0, :, 0].bool()  # (6890,) bool
        else:
            raise NotImplementedError
        
        with (pred_path).open("rb") as f:
            dict_pred = pickle.load(f)
        
        # 1. load pred results
        pred_verts = torch.from_numpy(dict_pred["pred_c_verts"]).cuda()
        pred_verts = apply_T_on_points(pred_verts[None], T_c2w[None])[0]  # (6890, 3)
        
        # 2. load gt
        gt_verts = np.load(gt_path)
        gt_verts = torch.from_numpy(gt_verts).float().cuda()
        # those are the old smplx logic
        # # gt_mesh = trimesh.load(gt_path, process=False)  # (10475, 3), in SMPLX
        # # gt_verts = torch.from_numpy(gt_mesh.vertices).float().cuda()
        # # gt_verts = smplx2smplh_def @ gt_verts
        
        # 3. load scene
        scan_name = meta["scan_name"]
        scene_name = meta["scene_key"][:-len(scan_name)-1]
        scene_path = Path(cfg.data.opt.test.root) / "sahmr_support" / "scene_info" / scene_name / f"{scan_name}_voxel_world.npy"
        scene = np.load(scene_path, allow_pickle=True).tolist()["points"]
        
        # 4. load pic
        image_path = Path(cfg.data.opt.test.root) / "sahmr_support" / "test_split" / meta["img_file"]
        
        # 5. visualize all the things
        wis3d.add_point_cloud(pred_verts, name="pred_verts")
        wis3d.add_point_cloud(gt_verts, name="gt_verts")
        wis3d.add_point_cloud(scene, name="scene")
        wis3d.add_image(str(image_path.absolute()), name="image")
        wis3d.set_scene_id(idx)
if __name__ == "__main__":
    main()