import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--out_name", type=str)
args = parser.parse_args()

# 设置根目录路径
root_dir = args.path

# 初始化一个大字典来存放所有文本描述
descriptions = {}

# 使用os.walk()遍历目录树
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        # 构建完整的文件路径
        file_path = os.path.join(subdir, filename)
        # 读取文件内容
        with open(file_path, "r") as file:
            content = file.read()
            try:
                # 将文件内容从字符串解析为字典
                data = json.loads(content)
                # 把需要的信息加入到大字典中
                key = file_path
                key = key.replace("smplerx_data", "smplerx_video")

                #### Idea400 first chatgpt results ####
                subset = key.split("/")[-2]
                subset_ = subset[:11]
                key = key.replace(subset, subset_)
                #######################################

                descriptions[key] = data["result"]
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")

# 指定最终的JSON文件名
output_file = os.path.join("./inputs/smplerx_data", args.out_name + ".json")
# 写入到JSON文件
with open(output_file, "w") as file:
    json.dump(descriptions, file, indent=4)
print(f"Save successfully at {output_file}")
