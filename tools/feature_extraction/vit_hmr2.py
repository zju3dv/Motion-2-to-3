from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hmr4d.network.hpe.hmr2 import load_hmr2, HMR2

# from hmr4d.dataset.hmr2vitfeat.rich import Dataset
from hmr4d.dataset.hmr2vitfeat.bedlam import Dataset

from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")


# Check if output file exists, this will skip already processed videos
# output_fn = Path("outputs/vit_feats/vitfeat_rich.pt")
output_fn = Path("outputs/vit_feats/vitfeat_bedlam.pt")

# TODO: Save according to the dataset


Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
outputs = {}
if output_fn.exists():
    print(f"Output file {output_fn} exists. Resumming")
    outputs = torch.load(output_fn)
vids_to_skip = list(outputs.keys())
save_frequency = 100

# Run HMR2.0 on all detected humans
dataset = Dataset(vids_to_skip=vids_to_skip)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: x[0])

# Setup HMR2.0 model
model: HMR2 = load_hmr2()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()

# Loop
for i, data in enumerate(tqdm(dataloader)):
    # Load data
    imgs, vid, meta = data
    F, _, H, W = imgs.shape  # (F, 3, H, W)
    imgs = imgs.to(device)

    # Run HMR2.0
    batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
    features = []
    for j in tqdm(range(0, F, batch_size), leave=False):
        imgs_batch = imgs[j : j + batch_size]

        with torch.no_grad():
            feature = model({"img": imgs_batch})
            features.append(feature.detach().cpu())

    features = torch.cat(features, dim=0)  # (F, 1024)

    # Save results
    assert "features" not in meta
    outputs[vid] = {"features": features, **meta}

    if i % save_frequency == 0 or i == len(dataloader) - 1:
        torch.save(outputs, output_fn)
