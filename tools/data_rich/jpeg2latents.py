import os
import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from pathlib import Path
from omegaconf import DictConfig

from hmr4d.dataset.rich.rich_motion_preprocess_latent import Dataset
from torch.utils.data import DataLoader

from transformers import CLIPVisionModelWithProjection
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize
import time
import torch.multiprocessing
from hmr4d.dataset.rich.rich_motion_preprocess_latent import Stage

torch.multiprocessing.set_sharing_strategy("file_system")


def stage1():
    # prepare dataset
    dataset = Dataset(split=split, stage=Stage.SAVE_IMG)
    data = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
    )

    imgkey_to_lurb = {}
    for batch in tqdm(data, total=len(data)):
        imgs = batch["img"]  # (B, 3, 224, 224)
        img_fns = batch["img_fn"]  # (B, )
        bbx_lurbs = batch["bbx_lurb"]  # (B, 4)
        img_keys = batch["img_key"]  # (B,)

        # Prepared cutted img
        imgs_np = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        imgs_np = imgs_np[..., ::-1]  # gbr to rgb, (B, 224, 224, 3)
        for b in range(len(imgs_np)):
            img_fn = str(img_fns[b])  # 'inputs/RICH/images_ds4/val/Pavallion_018_yoga1/cam_06/00875_06.jpeg'
            cutted_path = img_fn.replace("images_ds4", "hmr4d_support/cutted")
            os.makedirs(str(Path(cutted_path).parent), exist_ok=True)
            cv2.imwrite(cutted_path, imgs_np[b])

        # Append lurb
        bbx_lurbs_np = bbx_lurbs.cpu().numpy()
        for b, k in enumerate(img_keys):
            imgkey_to_lurb[k] = bbx_lurbs_np[b]

    lurb_out_fn = f"inputs/RICH/hmr4d_support/cutted/{split}_lurb.npy"
    np.save(lurb_out_fn, imgkey_to_lurb)


@torch.inference_mode()
def main():
    # Save cutted images and lurb, skip this step if already done
    if Path(f"inputs/RICH/hmr4d_support/cutted/{split}_lurb.npy").exists():
        print("Skipping Stage 1")
    else:
        stage1()
        print("Stage 1 finished")

    # prepare dataset
    dataset = Dataset(split=split, stage=Stage.LOAD_IMG)
    data = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=16,
    )

    # prepare clip things
    clip_pretrained_path = "inputs/checkpoints/huggingface/clip-vit-base-patch32"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_pretrained_path)
    image_encoder = image_encoder.cuda().eval()
    normalize_clip_img = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
    normalize_clip_img = normalize_clip_img.cuda()

    imgkey_to_saved_latent_info = {}
    for batch in tqdm(data, total=len(data)):
        imgs = batch["img"]  # (B, 3, 224, 224)
        img_keys = batch["img_key"]  # (B,)

        # Forward
        img_embeds = image_encoder(normalize_clip_img(imgs.cuda())).image_embeds  # (B, 512)
        img_embeds_np = img_embeds.cpu().numpy()
        for b, k in enumerate(img_keys):
            imgkey_to_saved_latent_info[k] = img_embeds_np[b]

    # save latents for whole seq_cam
    out_fn = f"inputs/RICH/hmr4d_support/cutted/{split}_img_embeds.npy"
    np.save(out_fn, imgkey_to_saved_latent_info)


if __name__ == "__main__":
    for split in ["val", "train", "test"]:
        tic = time.time()
        print("Start processing:", split)
        main()
        print("Finish processing:", split, "in", time.time() - tic, "seconds")

"""
Codes below are deprecated.
If the frames of each sequnce are equal, codes below will be fast.
"""

# import os
# import cv2
# import hydra
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from tqdm import tqdm
# from pathlib import Path
# from omegaconf import DictConfig
# from hmr4d.network.gmd.clip import CLIPLatentEncoder

# from hmr4d.dataset.rich.rich_motion_preprocess_latent import Dataset

# @hydra.main(version_base=None, config_path="../../hmr4d/configs", config_name="train")
# def main(cfg: DictConfig):
#     # alias params
#     args = cfg.model.network.args
#     args_clip = cfg.model.network.args_clip
#     args_unet = cfg.model.network.args_unet
#     enable_cfg = args.guidance_scale >= 1.0

#     # prepare dataset
#     dataset = Dataset(split=split)
#     data = DataLoader(
#         dataset,
#         batch_size=1,   # can't be more than 1 here! cause frames are different
#         shuffle=False,
#         num_workers=1,
#     )

#     # prepare clip things
#     clip = CLIPLatentEncoder(**args_clip)
#     clip.eval()
#     clip.requires_grad_(False)
#     clip.to("cuda")
#     clip_proj = nn.Linear(clip.clip_dim, args_unet.latent_dim)
#     clip_proj.eval()
#     clip_proj.requires_grad_(False)
#     clip_proj.to("cuda")

#     # iterate dataset
#     with tqdm(total=len(data)) as pbar:
#         pbar.set_description(f"Processing {split}")

#         for _, d in enumerate(data):
#             imgs      = d["prompt_imgs"]             # (B, frames, 3, 244, 244)
#             img_fns   = d["prompt_img_fns"]          # (frames, B)
#             bbx_lurbs = d["bbx_lurbs"]               # (B, frames, 4)
#             first_fid = d["first_fid"].cpu().numpy() # (B, )
#             # remove useless axis
#             # print(imgs.shape, bbx_lurbs.shape, len(img_fns[0]))
#             B = imgs.shape[0]
#             imgs = imgs.transpose(0, 1) # (frames, B, 3, 244, 244)
#             bbx_lurbs = bbx_lurbs.transpose(0, 1) # (frames, B, 4)
#             assert imgs.shape[0] == len(img_fns) and imgs.shape[0] == len(bbx_lurbs)

#             # prepare the directory
#             base_dir = [""] * B
#             for bid in range(B):
#                 parent_dir = str(Path(img_fns[0][bid]).parent)
#                 assert parent_dir == str(Path(img_fns[-1][bid]).parent)  # make sure the same parent dir
#                 base_dir[bid] = parent_dir.replace("images_ds4", "hmr4d_support/cutted")
#                 os.makedirs(base_dir[bid], exist_ok=True)

#             # save cutted img
#             latents = []
#             for i in range(len(img_fns)):
#                 img = imgs[i]   # (B, 3, 244, 244)

#                 # compute latent for single image
#                 prompt_latent = clip.encode_image_sequence(img[:, None].to("cuda"), enable_cfg=enable_cfg)
#                 prompt_latent = clip_proj(prompt_latent)  # (2*B, L=1, D)
#                 chunks = prompt_latent.chunk(2) # (2, B, 1, D)
#                 prompt_latent = torch.stack(chunks, dim=1) # (B, 2, 1, D)
#                 latents.append(prompt_latent.cpu().numpy()) # (frames, B, 2, 1, D)

#                 # save cutted image
#                 for bid in range(B):
#                     img_b = img[bid].permute(1, 2, 0) * 255
#                     img_b = img_b.cpu().numpy()
#                     img_b = img_b[..., ::-1]    # gbr to rgb
#                     cutted_path = img_fns[i][bid].replace("images_ds4", "hmr4d_support/cutted")
#                     cv2.imwrite(cutted_path, img_b)

#             # save latents for whole seq_cam
#             latents = np.stack(latents) # (frames, B, 2, 1, D)
#             for bid in range(B):
#                 latent_path = os.path.join(base_dir[bid], "latents.npy")
#                 np.save(latent_path, latents[:, bid])

#             # save lurbs for whole seq_cam
#             lurbs = bbx_lurbs.cpu().numpy() # (frames, B, 4)
#             for bid in range(B):
#                 lurbs_path = os.path.join(base_dir[bid], "lurbs.npy")
#                 np.save(lurbs_path, lurbs[:, bid])

#             # save a text to indicate the offset
#             for bid in range(B):
#                 offset_path = os.path.join(base_dir[bid], "offset.txt")
#                 with open(offset_path, "w") as f:
#                     f.write(str(first_fid[bid]))

#             pbar.update(1)

# if __name__ == "__main__":
#     for split in ["val", "train", "test"]:
#         main()
