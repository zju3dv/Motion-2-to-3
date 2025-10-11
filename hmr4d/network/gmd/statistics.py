from pathlib import Path
import numpy as np

data_dir = Path(__file__).parent / "data"

GMD_HMLVEC263 = {
    "mean": np.load(data_dir / "Mean_abs_3d.npy", allow_pickle=True),
    "std": np.load(data_dir / "Std_abs_3d.npy", allow_pickle=True),
}

GMD_EMPH_PROJ = np.load(data_dir / "rand_proj.npy", allow_pickle=True)
