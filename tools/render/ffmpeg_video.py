import os
import subprocess

# 定义主目录
BASE_DIR = "visualizations/blender_video"

# 定义 ffmpeg 命令模板
FFMPEG_CMD_TEMPLATE = "ffmpeg -framerate 30 -i {input_pattern} -c:v libx264 -crf 1 -preset slow -pix_fmt yuv420p {output_path} -y"

def convert_frames_to_videos(base_dir):
    # 遍历主目录下的所有 *_mesh_frames 文件夹
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.endswith("_mesh_frames"):
                folder_path = os.path.join(root, dir_name)
                parent_folder = os.path.dirname(folder_path)
                video_name = dir_name.replace("_mesh_frames", ".mp4")
                output_path = os.path.join(parent_folder, video_name)
                
                # 调试输出
                print(f"检测到文件夹：{folder_path}")
                print(f"生成视频路径：{output_path}")
                
                # 检查是否有帧文件
                frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_") and f.endswith(".png")]
                if frame_files:
                    print(f"检测到帧文件数量：{len(frame_files)}")
                    
                    # 构造 ffmpeg 命令
                    input_pattern = os.path.join(folder_path, "frame_%04d.png")
                    ffmpeg_cmd = FFMPEG_CMD_TEMPLATE.format(input_pattern=input_pattern, output_path=output_path)
                    
                    # 调用 ffmpeg
                    print(f"正在处理：{folder_path}")
                    subprocess.run(ffmpeg_cmd, shell=True, check=True)
                else:
                    print(f"跳过：{folder_path}（没有帧文件）")
    print("所有视频转换完成！")

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        print(f"目录不存在：{BASE_DIR}")
    else:
        convert_frames_to_videos(BASE_DIR)