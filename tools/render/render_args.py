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
import argparse

MODE = "video"
FOLDER = "visualizations/blender_video"
INPUT_MODE = "npy"
ON_FLOOR = False
DELETE = False
CAMERA_XY = None
CAMERA_ROT = None
SELECT_INDEX = None

# 过滤掉 Blender 自己的参数，保留 `--` 后的自定义参数
blender_args = sys.argv
if "--" in blender_args:
    idx = blender_args.index("--")
    custom_args = blender_args[idx + 1:]  # `--` 后的参数
else:
    custom_args = []  # 没有自定义参数

parser = argparse.ArgumentParser(description="处理命令行输入")

# 添加 `mode` 参数，限制值为 "video" 或 "sequence"
parser.add_argument(
    "--mode",
    type=str,
    choices=["video", "sequence"],
    required=True,
    help="选择模式: 'video' 或 'sequence'"
)

# 添加 `PATH` 参数，指定 mesh 的位置
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="指定 mesh 文件的路径"
)

# 添加 `camera_xy` 参数
parser.add_argument(
    "--camera-xy",
    type=float,
    nargs=2,
    required=False,
    default=None,
    help="相机位置的 XY 坐标，格式为: X Y"
)

# 添加 `camera_rot` 参数
parser.add_argument(
    "--camera-rot",
    type=float,
    nargs=3,
    required=False,
    default=None,
    help="相机旋转的角度，格式为: X Y Z"
)

# 解析参数
args = parser.parse_args(custom_args)
MODE = args.mode
PATH = args.path
if "motionclip" in PATH:
    CAMERA_XY = [-7.5, 0]
    CAMERA_ROT =[57, 0.0, 270]

if args.camera_xy:
    pass
    # CAMERA_XY = args.camera_xy
if args.camera_rot:
    pass
    # CAMERA_ROT = args.camera_rot


def render_cli() -> None:
    # parse options
    # output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type) , str(cfg.NAME)))
    # create logger
    # logger = create_logger(cfg, phase='render')

    if INPUT_MODE.lower() == "npy":
        out_path = PATH.replace("outputs", FOLDER)
        output_dir = Path(os.path.dirname(out_path))
        paths = [PATH]
        # print("xxx")
        # print("begin to render for{paths[0]}")
    elif INPUT_MODE.lower() == "dir":
        out_path = PATH.replace("outputs", FOLDER)
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
            # shutil.rmtree(frames_folder)
            # print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")


if __name__ == "__main__":
    render_cli()
