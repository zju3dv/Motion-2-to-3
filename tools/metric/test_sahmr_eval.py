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


def compute_error_accel(joints_gt, joints_pred):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    * modefied from VIBE: https://github.com/mkocabas/VIBE
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)
    return np.mean(normed, axis=1)


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
#               eval               #
# ================================ #


@monitor_process_wrapper
def get_data_loader(cfg: DictConfig) -> pl.LightningDataModule:
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    return datamodule.test_dataloader()


@hydra.main(version_base=None, config_path="../../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    Log.info(OmegaConf.to_yaml(cfg, resolve=False))
    dump_dir = Path(cfg.output_dir) / "pred_dump"
    assert dump_dir.exists(), "Run 'tools/test_sahmr_dump.py' to dump the results first!"

    # preparation
    data_loader = get_data_loader(cfg)
    test_dir = Path(cfg.data.opt.test.root) / "sahmr_support" / "test_split"
    sdfs = load_scene_sdfs(cfg)

    smplx2smplh_def_pth = Path(
        "inputs/models/model_transfer/smplx2smplh_deftrafo_setup.pkl"
    )  # TODO: should this path be in config?
    smplx2smplh_def = pickle.load(smplx2smplh_def_pth.open("rb"), encoding="latin1")
    smplx2smplh_def = np.array(smplx2smplh_def["mtx"].todense(), dtype=np.float32)[:, :10475]  # (6890, 10475)
    smplx2smplh_def = torch.from_numpy(smplx2smplh_def).float().cuda()

    H36M_J_regressor = np.load(
        "hmr4d/network/sahmr/sahmr/data/J_regressor_h36m_correct.npy",
        allow_pickle=True,  # TODO: should this path be in config?
    )  # (17, 6890)
    H36M_J_regressor = torch.from_numpy(H36M_J_regressor).float().cuda()
    H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

    GMPJPEs, GMPVEs, MPJPEs, MPVEs, PenEs, ConFEs = [], [], [], [], [], []
    joints_seq_pred = []
    joints_seq_gt = []
    for batch in tqdm(data_loader):
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

        # load gt
        gt_verts = np.load(gt_path)
        gt_verts = torch.from_numpy(gt_verts).float().cuda()
        # those are the old smplx logic
        # # gt_mesh = trimesh.load(gt_path, process=False)  # (10475, 3), in SMPLX
        # # gt_verts = torch.from_numpy(gt_mesh.vertices).float().cuda()
        # # gt_verts = smplx2smplh_def @ gt_verts

        # load pred
        with (pred_path).open("rb") as f:
            dict_pred = pickle.load(f)
        pred_verts = torch.from_numpy(dict_pred["pred_c_verts"]).cuda()
        pred_verts = apply_T_on_points(pred_verts[None], T_c2w[None])[0]  # (6890, 3)

        # ----- #
        gt_joints = H36M_J_regressor @ gt_verts
        gt_pelvis = gt_joints[0]
        gt_joints = gt_joints[H36M_J17_TO_J14]
        joints_seq_gt.append(gt_joints.cpu().tolist())
        # Log.debug(f"{H36M_J_regressor.shape}, {gt_verts.shape}, {gt_joints.shape}")

        pred_joints = H36M_J_regressor @ pred_verts
        pred_pelvis = pred_joints[0]
        pred_joints = pred_joints[H36M_J17_TO_J14]
        joints_seq_pred.append(pred_joints.cpu().tolist())
        # Log.debug(f"{H36M_J_regressor.shape}, {pred_verts.shape}, {pred_joints.shape}")

        q_gs_points = ((pred_verts - grid_min) / (grid_max - grid_min) * 2 - 1)[None]
        q_gs_sdfs = torch.nn.functional.grid_sample(
            sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
            q_gs_points[:, :, [2, 1, 0]].view(1, 1, 1, 6890, 3),
            padding_mode="border",
            align_corners=True,
        ).reshape(-1)
        PenE = q_gs_sdfs.abs() * (q_gs_sdfs < 0).float()
        ConFE = torch.cat([PenE[~hsc_mask], q_gs_sdfs.abs()[hsc_mask]])

        PenEs.append(to_np(PenE.sum()))
        ConFEs.append(to_np(ConFE.sum()))
        GMPJPEs.append(to_np(L2_error(gt_joints, pred_joints).mean() * 1000))
        GMPVEs.append(to_np(L2_error(gt_verts, pred_verts).mean() * 1000))
        MPJPEs.append(to_np(L2_error(gt_joints, pred_joints - pred_pelvis + gt_pelvis).mean() * 1000))
        MPVEs.append(to_np(L2_error(gt_verts, pred_verts - pred_pelvis + gt_pelvis).mean() * 1000))

    joints_seq_gt = np.array(joints_seq_gt)
    joints_seq_pred = np.array(joints_seq_pred)
    Accel = compute_error_accel(joints_seq_gt, joints_seq_pred)
    # fmt:off
    summarized_metrics = {
        "GMPJPE": np.mean(GMPJPEs),
        "MPJPE" : np.mean(MPJPEs),
        "Accel" : np.mean(Accel) * 1000,
        "GMPVE" : np.mean(GMPVEs),
        "MPVE"  : np.mean(MPVEs),
        "PenE"  : np.mean(PenEs),
        "ConFE" : np.mean(ConFEs),
    }
    # fmt:on

    for k, v in summarized_metrics.items():
        Log.info(f"{k}: {v:.2f}")
    Log.info("---------- ⬆️ ----------")


if __name__ == "__main__":
    main()
