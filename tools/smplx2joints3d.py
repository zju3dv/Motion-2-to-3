import os
import glob
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx

# fmt: off
OPT = {
    "rich_body_dirs"    : [Path(f"inputs/RICH/bodies/{x}_body") for x in ['train', 'val', 'test']],
    "output_dir"        : Path("inputs/RICH/hmr4d_support/joints3d")
}
gender_map = {"4": "male", "14": "male", "5": "male", "6": "male", "8": "male", "18": "female", "20": "male", "0": "male", "1": "male", "7": "male", "2": "male", "13": "female", "3": "male", "15": "male", "16": "female", "11": "male", "17": "female", "19": "female", "21": "female", "9": "female", "12": "female", "10": "female"}
# fmt: on

# Load SMPLX-SMPL
smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to_dense().cuda()
smpl_J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
print("smplx2smpl shape:", smplx2smpl.shape)


@torch.no_grad()
def get_joint3ds(body_model, smplx_param_fns):
    """Get smplx output, then convert to joint3ds."""
    # Load pkls
    body_params = []
    for fn in smplx_param_fns:
        with open(fn, "rb") as f:
            body_params.append(pickle.load(f))
    body_params_keys = body_params[0].keys()
    body_params = {k: np.concatenate([params[k] for params in body_params], axis=0) for k in body_params_keys}

    # Convert to torch, move to cuda
    body_params = {k: torch.from_numpy(v).cuda() for k, v in body_params.items()}

    # 4. get smpl from the model
    model_output = body_model(**body_params)
    # joints22 = model_output.joints.detach().cpu()[:, :22]  # first 22 joints equals to the joints of SMPL-22
    smplx_verts = model_output.vertices.detach()  # (N, 6890, 3)
    smpl_verts_converted = torch.matmul(smplx2smpl[None], smplx_verts)  # (N, 6890, 3)
    joints22 = torch.matmul(smpl_J_regressor[None], smpl_verts_converted)[:, :22]  # (N, 22, 3)
    joints22 = joints22.cpu()
    return joints22


if __name__ == "__main__":
    seq_dirs = []
    for root_dir in OPT["rich_body_dirs"]:
        seq_dirs.extend(sorted(root_dir.glob(f"*")))

    # Initialize SMPL-X body models
    body_models = {
        "male": make_smplx("rich-smplx", gender="male"),
        "female": make_smplx("rich-smplx", gender="female"),
    }
    body_models = {k: v.cuda() for k, v in body_models.items()}

    # For each sequence, get the joints3d (F, 22, 3)
    output = {}
    for seq_dir in tqdm(seq_dirs, desc="sequences"):
        seq_name = seq_dir.name
        # NOTE: we do not handle multiple people
        if len(seq_name.split("_")) > 3:
            continue
        subject_id = seq_name.split("_")[1]

        # Prepare target fns
        # |-- test_body/
        # |   |-- LectureHall_009_021_reparingprojector1/
        # |   |   |-- xxxxx/
        # |   |   |   |-- 009.pkl
        # |   |   |   |-- 021.pkl
        smplx_param_fns = list(sorted(seq_dir.glob(f"**/{subject_id}.pkl")))

        # Get joint3ds
        body_model = body_models[gender_map[str(int(subject_id))]]
        joints22 = get_joint3ds(body_model, smplx_param_fns)  # (F, 22, 3)

        frame_ids = [fn.parent.name for fn in smplx_param_fns]
        start_frame, end_frmame = int(frame_ids[0]), int(frame_ids[-1])
        output[seq_name] = (joints22.numpy(), start_frame, end_frmame)
        assert (end_frmame - start_frame + 1) == output[seq_name][0].shape[0]

    # Save
    os.makedirs(OPT["output_dir"], exist_ok=True)
    np.save(OPT["output_dir"] / "joints3d_smpl_v2.npy", output)
