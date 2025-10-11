import numpy as np
import torch
import copy
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .wis3d_utils import get_const_colors, color_schemes
from .skeleton_utils import SMPL_SKELETON, NBA_SKELETON, GYM_SKELETON, COLOR_NAMES, H36M_SKELETON
#import hmr4d.utils.matrix as matrix

import os
import subprocess
import shutil

def get_kinematic_chain(skeleton_type="nba"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    elif skeleton_type == "h36m":
        skeleton = H36M_SKELETON    
    else:
        raise NotImplementedError
    kinematic_chain = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["kinematic_chain"]
    ]
    return kinematic_chain


def get_skeleton_lines(skeleton_connections, ax):
    connections = []
    color_names = COLOR_NAMES
    for i, skel_con in enumerate(skeleton_connections):
        for _ in range(len(skel_con) - 1):
            c = np.array(color_schemes[color_names[i]][1]) / 255.0
            (line,) = ax.plot([], [], "k-", color=c)
            connections.append(line)
    return connections


def get_joints_color(pos, skeleton_type="nba"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    elif skeleton_type == "h36m":
        skeleton = H36M_SKELETON
    else:
        raise NotImplementedError
    color_names = COLOR_NAMES
    joints_category = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["joints_category"]
    ]
    joints_color = []
    J = pos.shape[-2]
    for i in range(J):
        for j, joints_ in enumerate(joints_category):
            if i in joints_:
                joints_color.append(color_schemes[color_names[j]][1])
                break
    joints_color = np.array(joints_color) / 255.0
    return joints_color


class plt_skeleton_animation:
    def __init__(self, pos, text="", skeleton_type="nba"):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions (x, y) and confidence (optional)
            text (str)
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if len(pos.shape) == 3:
            pos = pos[None]
        pos = copy.deepcopy(pos)
        pos[..., 1] *= -1
        self.pos = pos
        self.text = text
        self.skeleton_type = skeleton_type
        fig, ax = plt.subplots()
        fig.suptitle(text)
        self.fig = fig
        self.ax = ax
        joints_color = get_joints_color(pos, skeleton_type)
        points = []
        for i in range(pos.shape[-2]):
            points.append(plt.plot([], [], "o", color=joints_color[i])[0])
        self.points = points

        self.skeleton_connections = get_kinematic_chain(skeleton_type)
        self.connections = get_skeleton_lines(self.skeleton_connections, ax)

        self.paused = False
        self.current_frame = 0
        self.current_progress = pos.shape[0] - 1

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=pos.shape[-3],
            init_func=self.plt_init,
            blit=True,
            interval=33,
        )
        fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.show()

    def plt_init(self):
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            x_min, y_min, _ = self.pos.min(axis=(0, 1, 2))
            x_max, y_max, _ = self.pos.max(axis=(0, 1, 2))
        else:
            # x, y
            x_min, y_min = self.pos.min(axis=(0, 1, 2))
            x_max, y_max = self.pos.max(axis=(0, 1, 2))
        x_lim_min = x_min * 0.8 if x_min > 0 else x_min * 1.2
        x_lim_max = x_max * 0.8 if x_max < 0 else x_max * 1.2
        y_lim_min = y_min * 0.8 if y_min > 0 else y_min * 1.2
        y_lim_max = y_max * 0.8 if y_max < 0 else y_max * 1.2
        self.ax.set_xlim(x_lim_min, x_lim_max)
        self.ax.set_ylim(y_lim_min, y_lim_max)
        #plt.gca().invert_yaxis() # Added for inverse situation
        return *self.points, *self.connections

    def update(self, frame):
        p = self.current_progress
        x, y = self.pos[p, frame, :, 0], self.pos[p, frame, :, 1]
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            confidence = self.pos[p, frame, :, 2]
        else:
            # x, y
            confidence = np.ones_like(self.pos[p, frame, :, 0])
        for i in range(self.pos.shape[-2]):
            self.points[i].set_data(x[i], y[i])
            confidence[i] = min(confidence[i], 1.0)
            confidence[i] = max(confidence[i], 0.0)
            self.points[i].set_alpha(confidence[i])
            self.points[i].set_markersize(20 * confidence[i])
            # self.points[i].set_markersize(1)
        i = 0
        for skel_con in self.skeleton_connections:
            for j in range(len(skel_con) - 1):
                a = skel_con[j]
                b = skel_con[j + 1]
                self.connections[i].set_data([x[a], x[b]], [y[a], y[b]])
                i += 1
        return *self.points, *self.connections

    def on_key(self, event):
        if event.key == "p":
            if self.paused:
                self.animation.event_source.start()
            else:
                self.animation.event_source.stop()
            self.paused = ~self.paused
        if event.key == "u":
            self.current_progress = min(self.current_progress + 1, self.pos.shape[0] - 1)
        if event.key == "y":
            self.current_progress = max(self.current_progress - 1, 0)

class plt_skeleton_animation_mv:
    def __init__(self, pos, text="", skeleton_type="nba"):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions (x, y) and confidence (optional)
            text (str)
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if len(pos.shape) == 3:
            pos = pos[None]
        pos = copy.deepcopy(pos)
        pos[..., 1] *= -1
        self.pos = pos
        self.text = text
        self.skeleton_type = skeleton_type
        fig, ax = plt.subplots()
        fig.suptitle(text)
        self.fig = fig
        self.ax = ax
        joints_color = get_joints_color(pos, skeleton_type)
        points = []
        for i in range(pos.shape[-2]):
            points.append(plt.plot([], [], "o", color=joints_color[i])[0])
        self.points = points

        self.skeleton_connections = get_kinematic_chain(skeleton_type)
        self.connections = get_skeleton_lines(self.skeleton_connections, ax)

        self.paused = False
        self.current_frame = 0
        self.current_progress = pos.shape[0] - 1

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=pos.shape[-3],
            init_func=self.plt_init,
            blit=True,
            interval=33,
        )
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def plt_init(self):
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            x_min, y_min, _ = self.pos.min(axis=(0, 1, 2))
            x_max, y_max, _ = self.pos.max(axis=(0, 1, 2))
        else:
            # x, y
            x_min, y_min = self.pos.min(axis=(0, 1, 2))
            x_max, y_max = self.pos.max(axis=(0, 1, 2))
        x_lim_min = x_min * 0.8 if x_min > 0 else x_min * 1.2
        x_lim_max = x_max * 0.8 if x_max < 0 else x_max * 1.2
        y_lim_min = y_min * 0.8 if y_min > 0 else y_min * 1.2
        y_lim_max = y_max * 0.8 if y_max < 0 else y_max * 1.2
        self.ax.set_xlim(x_lim_min, x_lim_max)
        self.ax.set_ylim(y_lim_min, y_lim_max)
        #plt.gca().invert_yaxis() # Added for inverse situation
        return *self.points, *self.connections

    def update(self, frame):
        p = self.current_progress
        x, y = self.pos[p, frame, :, 0], self.pos[p, frame, :, 1]
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            confidence = self.pos[p, frame, :, 2]
        else:
            # x, y
            confidence = np.ones_like(self.pos[p, frame, :, 0])
        for i in range(self.pos.shape[-2]):
            self.points[i].set_data(x[i], y[i])
            confidence[i] = min(confidence[i], 1.0)
            confidence[i] = max(confidence[i], 0.0)
            self.points[i].set_alpha(confidence[i])
            self.points[i].set_markersize(20 * confidence[i])
            # self.points[i].set_markersize(1)
        i = 0
        for skel_con in self.skeleton_connections:
            for j in range(len(skel_con) - 1):
                a = skel_con[j]
                b = skel_con[j + 1]
                self.connections[i].set_data([x[a], x[b]], [y[a], y[b]])
                i += 1
        return *self.points, *self.connections

    def on_key(self, event):
        if event.key == "p":
            if self.paused:
                self.animation.event_source.start()
            else:
                self.animation.event_source.stop()
            self.paused = ~self.paused
        if event.key == "u":
            self.current_progress = min(self.current_progress + 1, self.pos.shape[0] - 1)
        if event.key == "y":
            self.current_progress = max(self.current_progress - 1, 0)


def visualize_2d_motion_as_video_mv(pred_mv, txt, index=0, save_dir='visualizations', fps=30):
    os.makedirs(save_dir, exist_ok=True)

    # index = torch.randperm(len(dataset))[:10]
    # index = [0]
    for i in range(1):
        print(txt)
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
        
        joints_color = get_joints_color(motion2d[0], "smpl")

        # 创建文件夹来保存图片
        image_dir = os.path.join(save_dir, f'{i:03}')
        os.makedirs(image_dir, exist_ok=True)
        for frame_idx in range(pred_mv.shape[1]):
            joints = motion2d[:, frame_idx, :, :] # 4 150 23 2
            # print(joints.shape)
        # for frame_idx, joints in enumerate(motion2d):
            # 创建空白帧
            frame = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

            # 缩放并偏移关节点以适应帧大小
            # print(joints.shape)
            # joints_scaled = (joints / 224 * 1024).astype(int)
            for v in range(len(pred_mv)):
                joints_0 = joints[v,:,:]
                if v==0:
                    joints_scaled = ((joints_0 * 256.) + [512.*0 + 256., 256.]).astype(int)
                elif v==1:
                    joints_scaled = ((joints_0 * 256.) + [512.*1 + 256., 256.]).astype(int)
                elif v==2:
                    joints_scaled = ((joints_0 * 256.) + [512.*0 + 256., 512.*1 + 256.]).astype(int)
                else:
                    joints_scaled = ((joints_0 * 256.) + [512.*1 + 256., 512.*1 + 256.]).astype(int)
                # print(joints)
                for j, joint in enumerate(joints_scaled):
                    c = deepcopy(joints_color[j] * 255)
                    # if j > 0 and j - 1 < mask.shape[1] and not mask[frame_idx, j - 1]:
                    #     c *= 0
                    #     s = 10
                    # else:
                    #     s = 5
                    c *= 0
                    s = 10
                    # rgb to bgr
                    c = [c[2], c[1], c[0]]
                    # print(joint[0])
                    # print((joint[0] + int(1024.*v), joint[1]))
                    cv2.circle(frame, (joint[0], joint[1]), s, c, -1)
                    # cv2.circle(frame, (joint[0] + int(1024.*v), joint[1]), s, c, -1)
                    
                # 绘制骨架连接
                for sk_i, skel_con in enumerate(skeleton_connections):
                    for j in range(len(skel_con) - 1):
                        a = skel_con[j]
                        b = skel_con[j + 1]
                        # pt1 = (joints_scaled[a][0]+int(1024.*v), joints_scaled[a][1])
                        # pt2 = (joints_scaled[b][0]+int(1024.*v), joints_scaled[b][1])
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
            text_position = (20, 30)  # 文本位置 (x, y)
            txt = txt.replace(".", ",")
            txt_arr = txt.split(',')
            for txt_s in range(len(txt_arr)):
                cv2.putText(frame, txt_arr[txt_s], (20, 30+30*txt_s), font, font_scale, font_color, thickness, line_type)
            cv2.putText(frame, "View 0", (10, 20+456), font, font_scale*3.0, font_color, thickness*2, line_type)
            cv2.putText(frame, "View 1", (10+512, 20+456), font, font_scale*3.0, font_color, thickness*2, line_type)
            cv2.putText(frame, "View 2", (10, 20+456+512), font, font_scale*3.0, font_color, thickness*2, line_type)
            cv2.putText(frame, "View 3", (10+512, 20+456+512), font, font_scale*3.0, font_color, thickness*2, line_type)
            # 保存每一帧为图片
            image_path = os.path.join(image_dir, f'frame_{frame_idx:04}.png')
            cv2.imwrite(image_path, frame)

        short_txt = txt[11:40].replace(" ", "_")
        # 使用 ffmpeg 将图片转换为视频
        video_path = os.path.join(save_dir, f'{index:05}_{short_txt}.mp4')
        ffmpeg_command = [
            'ffmpeg', '-y', '-r', str(fps), '-i', f'{image_dir}/frame_%04d.png',
            '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', video_path
        ]
        subprocess.run(ffmpeg_command)

        # 删除保存的图片文件夹
        # shutil.rmtree(image_dir)

        print(f"Video saved for {index:05}: {txt} to {video_path}")