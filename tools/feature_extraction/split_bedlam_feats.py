import torch
from pathlib import Path
import argparse
from tqdm import tqdm

# Do it on DPG10 server: python
parser = argparse.ArgumentParser()
parser.add_argument(
    "--full_vitfeat_pt",
    type=str,
    default="/home/shenzehong/Code/HMR-4D/inputs/bedlam/sm_support/vitfeat_bedlam.pt",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/shenzehong/Code/HMR-4D/inputs/bedlam/sm_support/vitfeats_30pt",
)
args = parser.parse_args()

# if input does not exist, exit
if not Path(args.full_vitfeat_pt).exists():
    print("Input file does not exist!")
    exit()

print("Loading....")
full_vitfeat_pt = Path(args.full_vitfeat_pt)
full_vitfeat = torch.load(full_vitfeat_pt)
# 37574 k-v pair
# key example: bedlam_data/bedlam_download/20221011_1_250_batch01hand_closeup_suburb_a/mp4/seq_000000.mp4-rp_christine_posed_021
# value example: 'features' (L, 1024), 'bbx_xys' (L, 3), 'img_hw'

# TODO: 20221024_10_100_batch01handhair_zoom_suburb_d.npz is missing

# Since this npz may be too large to load, we split it into 30 parts by part[2] in key. e.g. "20221011_1_250_batch01hand_closeup_suburb_a"
print("Spliting....")
vit_feat_splits = {}
for key, value in tqdm(full_vitfeat.items()):
    part = key.split("/")[2]
    if part not in vit_feat_splits:
        vit_feat_splits[part] = {}
    vit_feat_splits[part][key] = value

print("Saving....")
# For eace split, create a pt file under output_dir
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print("Saving to: ", output_dir)
for part, vit_feat_split in tqdm(vit_feat_splits.items()):
    output_pt = output_dir / f"{part}.pt"
    torch.save(vit_feat_split, output_pt)

# breakpoint()
