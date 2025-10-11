import torch
import cv2
import numpy as np
from hmr4d.dataset.wham_video.motion2d import Dataset as WHAM2D
from hmr4d.dataset.HumanML3D.motion2d import Dataset as HML2D
from hmr4d.dataset.ezi_video.motion2d import Dataset
from hmr4d.dataset.motionx.utils import normalize_keypoints_to_patch
from hmr4d.utils.plt_utils import get_joints_color, get_kinematic_chain
from hmr4d.utils.skeleton_utils import COLOR_NAMES
from hmr4d.utils.wis3d_utils import color_schemes
from copy import deepcopy
import os
import subprocess
import shutil

# USE_HML3D = False
# USE_HML3D = True 
THRESHOLD = 0.665

IS_WHAM = False

#### EGOEXO
# root = "inputs/ezi_data/egoexo_music_split_v2.pth" # music play instruments, do not move
root = "inputs/ezi_data/egoexo_climb_split_v0.pth" # climb
root = "inputs/ezi_data/egoexo_soccer_split_v1.pth" # soccer
# root = "inputs/ezi_data/egoexo_bike_split_v0.pth" # bike, do not move
# root = "inputs/ezi_data/egoexo_Dance_split_v1.pth" # Dance
# root = "inputs/ezi_data/egoexo_cook_split_v1.pth" # cook, do not move
root = "inputs/ezi_data/egoexo_basketball_split_v1.pth" # basketball
# root = "inputs/ezi_data/egoexo_omelet_split_v1.pth" # omelet, do not move
 
text_path = "inputs/ezi_data/egoexo_text_dict_all_v2_2.json"
text_path2 = "inputs/ezi_data/gpt35_egoexo_text_dict_all_v2.json"
threshold = 0.4
#########


##### MTPP 
# root = "inputs/ezi_data/motion_xpp_music_motion_v2_rt.pth" # music play instruments, do not move
# root = "inputs/ezi_data/motion_xpp_perform_motion_v0.pth" # perform
# root = "inputs/ezi_data/motion_xpp_kungfu_motion_v0.pth" # kungfu
# root = "inputs/ezi_data/motion_xpp_animation_motion_v0.pth" # animation
root = "inputs/ezi_data/motion_xpp_haa500_motion_v0.pth" # haa500
# root = "inputs/ezi_data/motion_xpp_humman_motion_v0.pth" # humman

text_path = "inputs/ezi_data/Motion_Xpp_v0.json"
# text_path2 = "inputs/ezi_data/gpt35_mtp_text_dict_all_v2.json"
threshold = 0.4
###########

##### WHAM
# IS_WHAM = True
# root = "inputs/wham_data/egoexo_dance_incam_motion_v0.pth" # egoexo dance
# text_path = "inputs/wham_data/egoexo_dance_text_v0.json"

# root = "inputs/wham_data/idea400_light_incam_motion_v0.pth" # idea400
# text_path = "inputs/wham_data/idea400_light_GT4V_v0.json"

# threshold = 0.4
###########


def visualize_2d_motion_as_video(dataset, save_dir='visualizations', fps=30):
    os.makedirs(save_dir, exist_ok=True)

    index = torch.randperm(len(dataset))[:10]
    for i in index:
        print(i)
        # 获取数据
        data = dataset._load_data(i)
        data = dataset._process_data(data, i)

        motion2d = data["gt_motion2d"]
        zero_root = torch.zeros_like(motion2d[:, :1, :])
        motion2d = torch.cat([zero_root, motion2d], dim=1)
        motion2d = motion2d.numpy()
        l = data["length"]
        motion2d = motion2d[:l]
        txt = data["text"]

        mask = (data["mask"] > THRESHOLD).float().numpy()
        mask = mask[:l]

        motion2d = normalize_keypoints_to_patch(motion2d, crop_size=224, inv=True)

        skeleton_connections = get_kinematic_chain("smpl")
        
        joints_color = get_joints_color(motion2d, "smpl")

        # 创建文件夹来保存图片
        image_dir = os.path.join(save_dir, f'{i:03}')
        os.makedirs(image_dir, exist_ok=True)

        for frame_idx, joints in enumerate(motion2d):
            # 创建空白帧
            frame = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

            # 缩放并偏移关节点以适应帧大小
            joints_scaled = (joints / 224 * 1024).astype(int)

            for j, joint in enumerate(joints_scaled):
                c = deepcopy(joints_color[j] * 255)
                if j > 0 and j - 1 < mask.shape[1] and not mask[frame_idx, j - 1]:
                    c *= 0
                    s = 10
                else:
                    s = 5
                # rgb to bgr
                c = [c[2], c[1], c[0]]
                cv2.circle(frame, (joint[0], joint[1]), s, c, -1)
                
            # 绘制骨架连接
            for sk_i, skel_con in enumerate(skeleton_connections):
                for j in range(len(skel_con) - 1):
                    a = skel_con[j]
                    b = skel_con[j + 1]
                    pt1 = (joints_scaled[a][0], joints_scaled[a][1])
                    pt2 = (joints_scaled[b][0], joints_scaled[b][1])
                    c = color_schemes[COLOR_NAMES[sk_i]][0]
                    c = deepcopy(c)
                    # rgb to bgr
                    c = [c[2], c[1], c[0]]

                    cv2.line(frame, pt1, pt2, c, 2)

            # 在每一帧上添加文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            thickness = 1
            line_type = cv2.LINE_AA
            text_position = (10, 20)  # 文本位置 (x, y)

            cv2.putText(frame, txt, text_position, font, font_scale, font_color, thickness, line_type)

            # 保存每一帧为图片
            image_path = os.path.join(image_dir, f'frame_{frame_idx:04}.png')
            cv2.imwrite(image_path, frame)

        # 使用 ffmpeg 将图片转换为视频
        video_path = os.path.join(save_dir, f'{i:03}.mp4')
        ffmpeg_command = [
            'ffmpeg', '-y', '-r', str(fps), '-i', f'{image_dir}/frame_%04d.png',
            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(ffmpeg_command)

        # 删除保存的图片文件夹
        shutil.rmtree(image_dir)

        print(f"Video saved for {i:03}: {txt} to {video_path}")

def visualize_2d_motion_as_video_mv(pred_mv, txt, save_dir='visualizations', fps=30):
    os.makedirs(save_dir, exist_ok=True)

    # index = torch.randperm(len(dataset))[:10]
    index = 1
    for i in index:
        print(i)
        # 获取数据
        # data = dataset._load_data(i)
        # data = dataset._process_data(data, i)

        # motion2d = data["gt_motion2d"]
        # zero_root = torch.zeros_like(motion2d[:, :1, :])
        # motion2d = torch.cat([zero_root, motion2d], dim=1)
        # motion2d = motion2d.numpy()
        # l = data["length"]
        # motion2d = motion2d[:l]
        # txt = data["text"]

        # mask = (data["mask"] > THRESHOLD).float().numpy()
        # mask = mask[:l]

        # motion2d = normalize_keypoints_to_patch(motion2d, crop_size=224, inv=True)
        motion2d = pred_mv.cpu().numpy()
        skeleton_connections = get_kinematic_chain("smpl")
        
        joints_color = get_joints_color(motion2d, "smpl")

        # 创建文件夹来保存图片
        image_dir = os.path.join(save_dir, f'{i:03}')
        os.makedirs(image_dir, exist_ok=True)

        for frame_idx, joints in enumerate(motion2d):
            # 创建空白帧
            frame = np.ones((4*1024, 1024, 3), dtype=np.uint8) * 255

            # 缩放并偏移关节点以适应帧大小
            joints_scaled = (joints / 224 * 1024).astype(int)

            for j, joint in enumerate(joints_scaled):
                c = deepcopy(joints_color[j] * 255)
                if j > 0 and j - 1 < mask.shape[1] and not mask[frame_idx, j - 1]:
                    c *= 0
                    s = 10
                else:
                    s = 5
                # rgb to bgr
                c = [c[2], c[1], c[0]]
                cv2.circle(frame, (joint[0], joint[1]), s, c, -1)
                
            # 绘制骨架连接
            for sk_i, skel_con in enumerate(skeleton_connections):
                for j in range(len(skel_con) - 1):
                    a = skel_con[j]
                    b = skel_con[j + 1]
                    pt1 = (joints_scaled[a][0], joints_scaled[a][1])
                    pt2 = (joints_scaled[b][0], joints_scaled[b][1])
                    c = color_schemes[COLOR_NAMES[sk_i]][0]
                    c = deepcopy(c)
                    # rgb to bgr
                    c = [c[2], c[1], c[0]]

                    cv2.line(frame, pt1, pt2, c, 2)

            # 在每一帧上添加文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            thickness = 1
            line_type = cv2.LINE_AA
            text_position = (10, 20)  # 文本位置 (x, y)

            cv2.putText(frame, txt, text_position, font, font_scale, font_color, thickness, line_type)

            # 保存每一帧为图片
            image_path = os.path.join(image_dir, f'frame_{frame_idx:04}.png')
            cv2.imwrite(image_path, frame)

        # 使用 ffmpeg 将图片转换为视频
        video_path = os.path.join(save_dir, f'{i:03}.mp4')
        ffmpeg_command = [
            'ffmpeg', '-y', '-r', str(fps), '-i', f'{image_dir}/frame_%04d.png',
            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(ffmpeg_command)

        # 删除保存的图片文件夹
        shutil.rmtree(image_dir)

        print(f"Video saved for {i:03}: {txt} to {video_path}")

# 使用示例
if USE_HML3D:
    dataset = HML2D(root="inputs/hml3d/joints3d.pth",
                split="train_val",
                distance=4.5,
                is_ignore_transl=True,
                is_root_next=True,
                # is_pinhole=True,
                is_pinhole=False,
                train_fps=30,
                )


elif IS_WHAM:
    dataset = WHAM2D(root=root,
                distance=4.5,
                is_ignore_transl=True,
                is_root_next=True,
                is_reproj=True,
                train_fps=30,
                text_path=text_path,
                )


else:
    dataset = Dataset(root=root, 
                  is_root_next=True,
                  text_path=text_path, 
                  text_path2=text_path2,
                  threshold_2d=threshold)



# 可视化示例
save_dir = 'outputs/visualizations/video'
if USE_HML3D:
    save_dir = 'outputs/visualizations/hml3d'
visualize_2d_motion_as_video(dataset, save_dir=save_dir, fps=30)