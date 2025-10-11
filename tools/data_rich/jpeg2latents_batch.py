import os
import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hmr4d.network.gmd.clip import CLIPLatentEncoder

from hmr4d.dataset.rich.rich_motion_preprocess_latent import Dataset

@hydra.main(version_base=None, config_path="../../hmr4d/configs", config_name="train")
def main(cfg: DictConfig):
    # alias params
    args = cfg.model.network.args
    args_clip = cfg.model.network.args_clip
    args_unet = cfg.model.network.args_unet
    enable_cfg = args.guidance_scale >= 1.0
        
    # prepare dataset
    dataset = Dataset(split=split)
    data = DataLoader(
        dataset,
        batch_size=1,   # can't be more than 1 here! cause frames are different
        shuffle=False,
        num_workers=1,
    )
    
    # prepare clip things
    clip = CLIPLatentEncoder(**args_clip)
    clip.eval()
    clip.requires_grad_(False)
    clip.to("cuda")
    clip_proj = nn.Linear(clip.clip_dim, args_unet.latent_dim)
    clip_proj.eval()
    clip_proj.requires_grad_(False)
    clip_proj.to("cuda")

    # iterate dataset
    with tqdm(total=len(data)) as pbar:
        pbar.set_description(f"Processing {split}")

        for _, d in enumerate(data):
            imgs      = d["prompt_imgs"]             # (B, frames, 3, 244, 244)
            img_fns   = d["prompt_img_fns"]          # (frames, B)
            bbx_lurbs = d["bbx_lurbs"]               # (B, frames, 4)
            first_fid = d["first_fid"].cpu().numpy() # (B, )
            # remove useless axis
            # print(imgs.shape, bbx_lurbs.shape, len(img_fns[0]))
            B = imgs.shape[0]
            imgs = imgs.transpose(0, 1) # (frames, B, 3, 244, 244)
            bbx_lurbs = bbx_lurbs.transpose(0, 1) # (frames, B, 4)
            assert imgs.shape[0] == len(img_fns) and imgs.shape[0] == len(bbx_lurbs)

            # prepare the directory
            base_dir = [""] * B
            for bid in range(B):
                parent_dir = str(Path(img_fns[0][bid]).parent)
                assert parent_dir == str(Path(img_fns[-1][bid]).parent)  # make sure the same parent dir
                base_dir[bid] = parent_dir.replace("images_ds4", "hmr4d_support/cutted")
                os.makedirs(base_dir[bid], exist_ok=True)

            # save cutted img
            latents = []
            for i in range(len(img_fns)):
                img = imgs[i]   # (B, 3, 244, 244)
                
                # compute latent for single image
                prompt_latent = clip.encode_image_sequence(img[:, None].to("cuda"), enable_cfg=enable_cfg)
                prompt_latent = clip_proj(prompt_latent)  # (2*B, L=1, D)
                chunks = prompt_latent.chunk(2) # (2, B, 1, D)
                prompt_latent = torch.stack(chunks, dim=1) # (B, 2, 1, D)
                latents.append(prompt_latent.cpu().numpy()) # (frames, B, 2, 1, D)

                # save cutted image
                for bid in range(B):
                    img_b = img[bid].permute(1, 2, 0) * 255
                    img_b = img_b.cpu().numpy()
                    img_b = img_b[..., ::-1]    # gbr to rgb
                    cutted_path = img_fns[i][bid].replace("images_ds4", "hmr4d_support/cutted")
                    cv2.imwrite(cutted_path, img_b)

            # save latents for whole seq_cam
            latents = np.stack(latents) # (frames, B, 2, 1, D)
            for bid in range(B):
                latent_path = os.path.join(base_dir[bid], "latents.npy")
                np.save(latent_path, latents[:, bid])
            
            # save lurbs for whole seq_cam
            lurbs = bbx_lurbs.cpu().numpy() # (frames, B, 4)
            for bid in range(B):
                lurbs_path = os.path.join(base_dir[bid], "lurbs.npy")
                np.save(lurbs_path, lurbs[:, bid])
            
            # save a text to indicate the offset
            for bid in range(B):
                offset_path = os.path.join(base_dir[bid], "offset.txt")
                with open(offset_path, "w") as f:
                    f.write(str(first_fid[bid]))
            
            pbar.update(1)

if __name__ == "__main__":
    for split in ["val", "train", "test"]:
        main()