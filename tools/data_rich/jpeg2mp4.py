# This file is used to preprocess the images from RICH to video.

import cv2
import numpy as np

from glob import glob
from pathlib import Path
from tqdm import tqdm
import imageio


OPT = {
    "rich_dirs"   : Path("inputs/RICH/images_ds4"),
    "output_dir"  : Path("inputs/RICH/hmr4d_support/video"),
    "fps"         : 30,
    "frame_suffix": "jpeg",
}

def get_video_list():
    """
    Glob all the video folders and flatten to a list.
    """
    ret = sorted(glob(str(OPT["rich_dirs"]) + "/*/*/*"))
    return ret

def record(v:str):
    """
    Glob all the frames of the video and save to a video.
    Also, notes the meta data of the video.
    """
    # 1. fetch all the frames
    read_pattern = v + "/*." + OPT["frame_suffix"]
    frames = sorted(glob(read_pattern))
    
    # 2. get the meta data of the frames
    h, w = cv2.imread(frames[0]).shape[:2]
    fid_start = int(frames[0].split("/")[-1].split("_")[0])
    fid_end   = int(frames[-1].split("/")[-1].split("_")[0])
    
    # 3. convert the images to video
    output_path = OPT["output_dir"] / Path(v).relative_to(OPT["rich_dirs"])
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path / "video.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(output_path, fourcc, OPT["fps"], (w, h))
    # for f in tqdm(frames, desc="Converting to video..."):
    #     video.write(cv2.imread(f))
    # video.release()
    images = []
    for f in tqdm(frames, desc="Converting to video..."):
        image = imageio.imread(f)
        # make the width and length divisible by 16,
        # by directly throwing away the last few pixels, which does not affect the image intrinsic
        h, w = image.shape[:2]
        image = image[:h//16*16, :w//16*16]
        images.append(image)
    imageio.mimsave(output_path, images, fps=OPT["fps"], quality=6)
    with open(output_path.replace("video.mp4", "meta.txt"), "w") as f:
        f.write(" ".join([str(fid_start), str(fid_end)]))

if __name__ == "__main__":
    video_list = get_video_list()
    for v in tqdm(video_list):
        print("Processing video: ", v)
        record(v)