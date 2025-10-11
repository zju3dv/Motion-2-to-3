import numpy as np
import torch
from time import time
from pathlib import Path
from tqdm import tqdm


def time_ellapse(func):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} took {time() - start} seconds")
        return res

    return wrapper


@time_ellapse
def load_pt(pt_fp):
    return torch.load(pt_fp)


@time_ellapse
def load_npz(npz_fp):
    return np.load(npz_fp, allow_pickle=True)


if __name__ == "__main__":
    # Input
    pt_folder = "inputs/bedlam/sm_support/vitfeats_30pt"
    pt_names = [fn.name for fn in Path(pt_folder).glob("*.pt")]

    # Output
    output_npz_folder = f"{pt_folder}_npz"
    Path(output_npz_folder).mkdir(exist_ok=True, parents=True)

    for pt_name in tqdm(pt_names):
        pt_fp = Path(pt_folder) / pt_name
        data = load_pt(pt_fp)  # fully loaded to RAM, which is not good for large dataset

        npz_fp = str(Path(output_npz_folder) / pt_name.replace(".pt", ".npz"))
        np.savez(npz_fp, **data)

    # # Check loading:
    # key = "bedlam_data/bedlam_download/20221019_1_250_highbmihand_closeup_suburb_c/mp4/seq_000248.mp4-male_40_nl_6353"
    # real_data = npzfile[key].item()
