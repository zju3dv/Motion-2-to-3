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

def smpl2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    // SMPL
    {0,  "Pelvis"},
    {1, "L_Hip"},
    {2, "R_Hip"},
    {3, "Spine1"},
    {4, "L_Knee"},
    {5, "R_Knee"},
    {6, "Spine2"},
    {7, "L_Ankle"},
    {8, "R_Ankle"},
    {9, "Spine3"},
    {10, "L_Foot"},
    {11, "R_Foot"},
    {12, "Neck"},
    {13, "L_Collar"},
    {14, "R_Collar"},
    {15, "Head"},
    {16, "L_Shoulder"},
    {17, "R_Shoulder"},
    {18, "L_Elbow"},
    {19, "R_Elbow"},
    {20, "L_Wrist"},
    {21, "R_Wrist"},
    {22, "L_Hand"},
    {23, "R_Hand"},
    '''
    T, V, C = x.shape
    y = torch.zeros((T,17,C))
    y[:,0,:] = x[:,0,:] # hip
    y[:,1,:] = x[:,2,:] # R_Hip
    y[:,2,:] = x[:,5,:] # R_knee
    y[:,3,:] = x[:,8,:] # R_ankle
    y[:,4,:] = x[:,1,:] # L_Hip
    y[:,5,:] = x[:,4,:] # L_knee
    y[:,6,:] = x[:,7,:] # L_ankle
    y[:,7,:] = (x[:,3,:] + x[:,6,:]) * 0.5 # spine1
    y[:,8,:] = x[:,12,:] # neck
    y[:,9,:] = (x[:,12,:] + x[:, 15, :]) * 0.5 # nose
    y[:,10,:] = x[:,15,:] # head
    y[:,11,:] = x[:,16,:] # L_Shoulder
    y[:,12,:] = x[:,18,:] # L_Elbow
    y[:,13,:] = x[:,20,:] # L_Wrist
    y[:,14,:] = x[:,17,:] # R_Shoulder
    y[:,15,:] = x[:,19,:] # R_Elbow
    y[:,16,:] = x[:,21,:] # R_Wrist
    return y

def visualize_2d_motion_as_video(all_data, save_dir='visualizations', fps=30):
    os.makedirs(save_dir, exist_ok=True)


    for i, data in enumerate(all_data):
        data = torch.load(data)

        motion2d = data["pred"]
        motion2d = smpl2h36m(motion2d)
        l = data["length"]
        motion2d = motion2d[:l]
        txt = data["text"]

        mask = torch.ones_like(motion2d[..., :1])

        motion2d = motion2d.numpy()

        motion2d = normalize_keypoints_to_patch(motion2d, crop_size=224, inv=True)

        skeleton_connections = get_kinematic_chain("h36m")
        
        joints_color = get_joints_color(motion2d, "h36m")

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

        break


# 可视化示例
save_dir = 'visualizations/h36m'
data_path = "./outputs/dumped_2donly_kungfu"
data = os.listdir(data_path)
data.sort()
data = [os.path.join(data_path, d) for d in data]
visualize_2d_motion_as_video(data, save_dir=save_dir, fps=30)