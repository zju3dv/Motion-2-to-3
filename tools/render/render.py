import os
import random
import shutil
import sys
from pathlib import Path

try:
    import bpy

    sys.path.append(os.path.dirname(bpy.data.filepath))
    sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.9/site-packages"))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )

from hmr4d.utils.render.joints import smplh_to_mmm_scaling_factor

MODE = "video"
MODE = "sequence"
FOLDER = "visualizations/blender_render"

INPUT_MODE = "npy"
ON_FLOOR = False
DELETE = False
CAMERA_XY = None
CAMERA_ROT = None

### teaser
# MODE = "sequence"
# MODEL = "singlemodel"
# SUBSET = "teaser"
# INPUT_MODE = "npy"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0019_mesh.npy"
# CAMERA_XY = [-7.1, -3.6]
# CAMERA_ROT =[61, 0.0, -68] # X and Z, and set Y as 0
# SELECT_INDEX = [30, 45, 90] # 19 handstand

# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0036_mesh.npy"
# CAMERA_XY = [-2.5, 11.4]
# CAMERA_ROT =[64, 0.0, -158] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 50, 82, 110, 136] # 36 juggle

# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0007_mesh.npy"
# CAMERA_XY = [-0.7, -11]
# CAMERA_ROT =[64, 0.0, -11] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 20, 40, 65, 120] # 7, crawling
############


#### Our basketball
# MODE = "frame"
# MODE = "sequence"
# MODEL = "singlemodel"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0005_mesh.npy"
# CAMERA_XY = [-7.1, -3.6]
# CAMERA_ROT =[61, 0.0, -68] # X and Z, and set Y as 0
# SELECT_INDEX = [15, 30, 45, 60, 70] # basketball005
##############


###### MDM basketball
# MODE = "sequence"
# # MODE = "frame"
# MODEL = "mdm"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0019_mesh.npy"
# # CAMERA_XY = [4.9, 5.6]
# # CAMERA_ROT =[56, 0.0, 140] # X and Z, and set Y as 0
# SELECT_INDEX = [30, 60, 90, 120, 145]
##############

##### MotionCLIP
# MODE = "sequence"
# # MODE = "frame"
# MODEL = "motionclip"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0000_mesh.npy"
# CAMERA_XY = [-6.7, 0]
# CAMERA_ROT =[54, 0.0, 270] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 30, 70, 100, 145]
#######

##### MotionBERT
# MODE = "sequence"
# MODE = "frame"
# MODEL = "motionbert"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0000_mesh.npy"
# CAMERA_XY = [6.1, 5.1]
# CAMERA_ROT =[57, 0.0, -230] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 48, 70, 100, 130]
#######

##### AvatarCLIP
# MODE = "sequence"
# MODE = "frame"
# MODEL = "avatarclip"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0000_mesh.npy"
# CAMERA_XY = [7.0, 0]
# CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 12, 24, 36, 48, 59]
#######

##### MAS
# MODE = "sequence"
# # MODE = "frame"
# MODEL = "mas"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0005_mesh.npy"
# # CAMERA_XY = [7.0, 0]
# # CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 30, 55, 90, 120]
#######

##### MLD
# MODE = "sequence"
# MODEL = "mld"
# SUBSET = "comp_basketball"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0003_mesh.npy"
# # CAMERA_XY = [7.0, 0]
# # CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 110, 120, 130, 149]
#######

####  COMP_2
# MODE = "sequence"
# MODEL = "single"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0926_mesh.npy"
# CAMERA_XY = [-2.5, -8.0]
# CAMERA_ROT =[60, 0.0, -20] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 30, 180, 290] # crawl and stand

# MODE = "sequence"
# MODEL = "mas"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0926_mesh.npy"
# # CAMERA_XY = [7.0, 0]
# # CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 59, 120, 180, 240]

# MODE = "sequence"
# MODEL = "mdm"
# SUBSET = "comp_2"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0000_mesh.npy"
# # CAMERA_XY = [7.0, 0]
# # CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 59, 120, 180, 240]

# MODE = "sequence"
# MODEL = "mld"
# SUBSET = "comp_2"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0002_mesh.npy"
# # CAMERA_XY = [7.0, 0]
# # CAMERA_ROT =[54, 0.0, 90] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 59, 120, 160, 292]

# MODE = "sequence"
# MODEL = "motionclip"
# SUBSET = "comp_2"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0000_mesh.npy"
# CAMERA_XY = [-6.7, 0]
# CAMERA_ROT =[54, 0.0, 270] # X and Z, and set Y as 0
# SELECT_INDEX = [0, 59, 120, 160, 292]

#### Ablation
## 1
# MODE = "sequence"
# MODEL = "single_fid308"
# MODEL = "single_cb"
# MODEL = "single_scratch"
# MODEL = "single_3view"
# MODEL = "single_5view"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/3926_mesh.npy"
# SELECT_INDEX = [0, 40, 80, 120, 180]
## 2
MODE = "sequence"
MODEL = "single_fid308"
MODEL = "single_cb"
MODEL = "single_scratch"
MODEL = "single_3view"
MODEL = "single_5view"
SUBSET = "t2m"
PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/3212_mesh.npy"
SELECT_INDEX = [0, 40, 70, 90, 120, 150, 250]
######

### comp_3
# ours
# MODE = "sequence"
# MODEL = "singlemodel"
# SUBSET = "userstudy"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0001_mesh.npy"
# SELECT_INDEX = [0, 80, 104, 160, 180]

# # mld
# MODE = "sequence"
# MODEL = "mld"
# SUBSET = "userstudy"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0001_mesh.npy"
# SELECT_INDEX = [0, 30, 80, 125, 196]

# # mdm
# MODE = "sequence"
# MODEL = "mdm"
# SUBSET = "userstudy"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0001_mesh.npy"
# SELECT_INDEX = [0, 30, 80, 125, 196]

# # motionclip
# MODE = "sequence"
# MODEL = "motionclip"
# SUBSET = "userstudy"
# CAMERA_XY = [-7.5, 0]
# CAMERA_ROT =[57, 0.0, 270] # X and Z, and set Y as 0
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0001_mesh.npy"
# SELECT_INDEX = [0, 30, 80, 125, 196]
###

### use2d
## pick up
# our 
# MODE = "sequence"
# MODEL = "single_fid308"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0003_mesh.npy"
# SELECT_INDEX = [0, 47, 67, 85, 130]

# 2d condition
# MODE = "sequence"
# MODEL = "ours"
# SUBSET = "t2m"
# CAMERA_XY = [-2.4, -8.5]
# CAMERA_ROT =[60, 0.0, -22] # X and Z, and set Y as 0
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0003_mesh.npy"
# SELECT_INDEX = [0, 22, 47, 67, 130]

# mas
# MODE = "sequence"
# MODEL = "mas"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0003_mesh.npy"
# SELECT_INDEX = [0, 22, 47, 67, 130]

# motionbert
# MODE = "sequence"
# MODEL = "motionbert"
# SUBSET = "t2m"
# CAMERA_XY = [7.3, 6.5]
# CAMERA_ROT =[62, 0.0, 134] # X and Z, and set Y as 0
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0003_mesh.npy"
# SELECT_INDEX = [0, 22, 47, 67, 130]
####
### walk backward and agian
# MODE = "sequence"
# MODEL = "single_fid308"
# SUBSET = "t2m"
# CAMERA_XY = [7.4, -7.3]
# CAMERA_ROT =[65, 0.0, 46.7] 
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0211_mesh.npy"
# SELECT_INDEX = [0, 58, 105, 144, 196, 237, 284]

# others methods
# MODE = "sequence"
# MODEL = "ours"
# MODEL = "mas"
# MODEL = "motionbert"
# SUBSET = "t2m"
# PATH = f"./outputs/dumped_{SUBSET}_{MODEL}_mesh/0211_mesh.npy"
# SELECT_INDEX = [0, 58, 105, 144, 196, 237, 284] # ours, mas
# SELECT_INDEX = [0, 58, 105, 144, 196] # motionbert

########

def render_cli() -> None:
    # parse options
    # output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type) , str(cfg.NAME)))
    # create logger
    # logger = create_logger(cfg, phase='render')

    if INPUT_MODE.lower() == "npy":
        out_path = PATH.replace("outputs", "visualizations/blender_render")
        output_dir = Path(os.path.dirname(out_path))
        paths = [PATH]
        # print("xxx")
        # print("begin to render for{paths[0]}")
    elif INPUT_MODE.lower() == "dir":
        out_path = PATH.replace("outputs", "visualizations/blender_render")
        output_dir = Path(out_path)
        paths = []
        # file_list = os.listdir(cfg.RENDER.DIR)
        # random begin for parallel
        file_list = os.listdir(PATH)
        begin_id = random.randrange(0, len(file_list))
        file_list = file_list[begin_id:]+file_list[:begin_id]

        # render mesh npy first
        for item in file_list:
            if item.endswith("_mesh.npy"):
                paths.append(os.path.join(PATH, item))

        # then render other npy
        for item in file_list:
            if item.endswith(".npy") and not item.endswith("_mesh.npy"):
                paths.append(os.path.join(PATH, item))

        print(f"begin to render for {paths[0]}")

    import numpy as np

    from hmr4d.utils.render.blender import render
    from hmr4d.utils.render.blender.tools import mesh_detect
    from hmr4d.utils.render.video import Video
    init = True
    for path in paths:
        # check existed mp4 or under rendering
        if MODE == "video":
            if os.path.exists(path.replace(".npy", ".mp4")) or os.path.exists(path.replace(".npy", "_frames")):
                print(f"npy is rendered or under rendering {path}")
                continue
        else:
            # check existed png
            if os.path.exists(path.replace(".npy", ".png")):
                print(f"npy is rendered or under rendering {path}")
                continue

        if MODE == "video":
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", "_frames").split('/')[-1])
            os.makedirs(frames_folder, exist_ok=True)
        else:
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", ".png").split('/')[-1])

        try:
            data = np.load(path)
            # if cfg.RENDER.JOINT_TYPE.lower() == "humanml3d":
            if True:
                is_mesh = mesh_detect(data)
                if not is_mesh:
                    data = data * smplh_to_mmm_scaling_factor
        except FileNotFoundError:
            print(f"{path} not found")
            continue

        if MODE == "video":
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", "_frames").split("/")[-1]
            )
        else:
            frames_folder = os.path.join(
                output_dir, path.replace(".npy", ".png").split("/")[-1]
            )
        print(frames_folder)
        # raise NotImplementedError

        out = render(
            data,
            frames_folder,
            canonicalize=True,
            exact_frame=0.5,
            num=8,
            mode=MODE,
            faces_path="./hmr4d/utils/render/deps/smplh/smplh.faces",
            downsample=False,
            always_on_floor=ON_FLOOR,
            # oldrender=True,
            oldrender=False,
            jointstype="humanml3d",
            res="high",
            init=init,
            gt=False,
            select_index=SELECT_INDEX,
            camera_xy=CAMERA_XY,
            camera_rot=CAMERA_ROT,
            delete=DELETE,
            accelerator="gpu",
            device=[0],
        )

        init = False

        if MODE == "video":
            if ON_FLOOR:
                frames_folder += "_of"
            if False:
                video = Video(frames_folder, fps=30)
            else:
                video = Video(frames_folder, fps=30)

            vid_path = frames_folder.replace("_frames", ".mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")


if __name__ == "__main__":
    render_cli()
