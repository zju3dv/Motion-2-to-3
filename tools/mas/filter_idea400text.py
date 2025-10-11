import numpy as np
import torch
import os
import codecs as cs
import json



def load_motionx_text(base_motion_path, base_text_path):
    total_text = []
    seqs_names = []
    
    # 遍历motion数据的子集文件夹
    for subset in os.listdir(base_motion_path):
        motion_subset_path = os.path.join(base_motion_path, subset)
        text_subset_path = os.path.join(base_text_path, subset)

        if os.path.isdir(motion_subset_path) and os.path.isdir(text_subset_path):
            # 遍历每个subset中的.npy文件
            for file in os.listdir(motion_subset_path):
                if file.endswith('.npy'):
                    motion_file_path = os.path.join(motion_subset_path, file)
                    text_file_path = os.path.join(text_subset_path, file.replace('.npy', '.txt'))

                    # 读取motion和text
                    if os.path.exists(text_file_path):
                        motion = np.load(motion_file_path)
                        if motion.shape[0] < 60:
                            continue
                        if motion.shape[0] > 300:
                            continue
                        with open(text_file_path, 'r') as f:
                            text = f.read().strip()
                        if "complete sentence" in text:
                            continue
                        if "The subject" in text:
                            continue
                        if "Output: " in text:
                            continue
                        if "difficult" in text:
                            continue
                        if "illegal" in text:
                            continue
                        if "##Ou" in text:
                            continue
                        if "Group" in text:
                            continue
                        if "group" in text:
                            continue
                        if "appropriate" in text:
                            continue
                        if "clear" in text:
                            continue
                        if "possible" in text:
                            continue
                        # if "sit" in text:
                        #     continue
                        if "mistake" in text:
                            continue
                        if "No one" in text:
                            continue
                        if text not in total_text:
                            total_text.append(text)
                            # save subset and seq_name
                            seqs_names.append(f"{subset}/{file[:-4]}")
                        

                        # 存储在字典中

    return total_text, seqs_names


def load_hml3d_text():
    split_file = f"./inputs/hml3d/test.txt"
    txt_path = f"./inputs/hml3d/texts"

    test_text = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            seq_name = line.strip()
            with cs.open(os.path.join(txt_path, seq_name + ".txt")) as text_f:
                for text_line in text_f.readlines():
                    line_split = text_line.strip().split("#")
                    caption = line_split[0]
                    test_text.append(caption)
    return test_text



subset = "idea400"
subset = "perform"
subset = "HAA500"
# subset = "kungfu"
# subset = "Dance" # egoexo 2d
subset = "dance" # egoexo wham
# subset = "soccer" # egoexo 2d
# subset = "basketball" # egoexo 2d

# remeber to run `pip install -U sentence-transformers` first

if subset == "idea400":
    train_txt_path = "inputs/wham_data/idea400_light_GT4V_v0.json"
    with open(train_txt_path, "r") as file:
        train_text = json.load(file)
else:
    data_path = f"inputs/ezi_data/motion_xpp_{subset.lower()}_motion_v0.pth"
    train_txt_path = "inputs/ezi_data/Motion_Xpp_v0.json"
    if os.path.exists(data_path):
        data = torch.load(data_path)
        with open(train_txt_path, "r") as file:
            all_train_text = json.load(file)
        train_text = {}
        for k in data.keys():
            if data[k]["name"] in all_train_text.keys():
                train_text[data[k]["name"]] = all_train_text[data[k]["name"]]
    else:
        data_path = f"inputs/ezi_data/egoexo_{subset}_split_v1.pth"
        train_txt_path = "inputs/ezi_data/egoexo_text_dict_all_v2_2.json"
        # train_txt_path = "inputs/ezi_data/gpt35_egoexo_text_dict_all_v2.json" # gpt35
        if os.path.exists(data_path):
            data = torch.load(data_path)
            with open(train_txt_path, "r") as file:
                all_train_text = json.load(file)
            train_text = {}
            for k in data.keys():
                name = data[k]["name"]
                if  '/' in name and "takes/" + name.split("/")[-2] in all_train_text.keys():
                    train_text[name] = all_train_text["takes/" +name.split("/")[-2]]
        else:
            data_path = f"inputs/wham_data/egoexo_{subset}_incam_motion_v0.pth"
            train_text_path = f"inputs/wham_data/egoexo_{subset}_text_v0.json"
            data = torch.load(data_path)
            with open(train_txt_path, "r") as file:
                all_train_text = json.load(file)
            train_text = all_train_text
        
printed_text = []
for k, v in train_text.items():
    if v not in printed_text:
        print(v)
        printed_text.append(v)
    # if "juggle" in v and v not in printed_text:
    #     print(v)
    #     printed_text.append(v)
    
test_motion_path =f"inputs/motionx/motion_data/smplx_322/{subset}" 
test_text_path = f"inputs/motionx/motionx_seq_text_v1.1/{subset}"
test_text, test_seq_names = load_motionx_text(test_motion_path, test_text_path)
exist_train_text = []

for key in train_text.keys():
    train_text_ = train_text[key]
    for test_text_ in test_text:
        if train_text_ == test_text_ and train_text_ not in exist_train_text:
            exist_train_text.append(train_text_)
            break
# remove exist text in test_text
for i in range(len(test_text)):
    if test_text[i] in exist_train_text:
        test_text[i] = None
print("Remove exist text in test_text: ", len(exist_train_text))

test_seq_names = [seq_name for seq_name, text in zip(test_seq_names, test_text) if text is not None]
test_text = [text for text in test_text if text is not None]



hml3d_test_text = load_hml3d_text()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

idea400_emb = model.encode(test_text)
hml3d_emb = model.encode(hml3d_test_text)
idea400_emb_norm = idea400_emb / np.linalg.norm(idea400_emb, axis=-1, keepdims=True)
hml3d_emb_norm = hml3d_emb / np.linalg.norm(hml3d_emb, axis=-1, keepdims=True)
cosine_similarity = np.dot(idea400_emb_norm, hml3d_emb_norm.T)
max_cosine_similarity = np.max(cosine_similarity, axis=-1)
# mask = max_cosine_similarity < 0.45 # follow Plan poseture and go
mask = max_cosine_similarity < 1.0 
# select idea400 text that is not similar to hml3d text
test_text = [text for text, m in zip(test_text, mask) if m]
test_seq_names = [seq_name for seq_name, m in zip(test_seq_names, mask) if m]
print(f"Select {len(test_seq_names)} sequences")

# save test_seq names
with open(f"inputs/motionx/motionx_seq_text_v1.1/{subset}_test_seq_names.json", "w") as f:
    json.dump(test_seq_names, f, indent=4)


import ipdb;ipdb.set_trace()

