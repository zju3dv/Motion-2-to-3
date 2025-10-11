import numpy as np
import torch
import os
import codecs as cs
from sklearn.cluster import KMeans
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.dataset.wham_video.utils import smpl_fk
import json

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



def load_motionx_text(base_motion_path, base_text_path):
    total_text = []
    seqs_names = []
    all_motion = []
    
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
                        if "##" in text:
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
                            all_motion.append(motion)
                        

    return total_text, seqs_names, all_motion


# subsets = ["idea400", "kungfu", "animation", "perform"]
subsets = ["idea400", "kungfu", "HAA500", "perform", "animation"]
subsets = ["idea400", "kungfu", "HAA500"]
subsets = ["idea400", "HAA500"]
subsets = ["idea400"] # ours bad
subsets = ["HAA500"] # ours bad
subsets = ["kungfu"] # ours bad
subsets = ["perform"] # ours bad
subsets = ["animation"] # ours_fid=18.142, 0.473, 8.183, mdm=19.503, 0.326, 6.908
subsets = ["dance"] # , mdm=23.471, 0.292, 8.140 NOTE:too small
subsets = ["humman"] # 19.024, 0.352, 7.773, mdm=12.438, 0.352, 7.872
subsets = ["music"] # 13.123, 0.172, 5.207, mdm=9.264, 0.192, 5.752
subsets = ["game_motion"] # 7.883, 0.262, 6.139, mdm=10.222, 0.298, 7.306
subsets = ["aist"] # 22.342, 0.228, 3.176, mdm = 31.247, 0.225, 4.371
subsets = ["animation", "game_motion", "aist"]
# subsets = ["idea400", "game_motion", "aist"]
subsets = ["animation", "aist"]
# subsets = ["idea400", "kungfu", "HAA500", "perform"]
subsets = ["animation", "humman", "game_motion", "aist"]
# FID, precision-top30, precision-top3, diversity, R_size=8
subsets = ["kungfu", "HAA500", "perform", "animation", "aist"]
# subsets = ["kungfu", "aist", "animation"]
# subsets = ["kungfu", "HAA500", "aist", "animation"]
# subsets = ["idea400"]



def select_subset(subset):
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
            train_text = {}


    test_motion_path =f"inputs/motionx/motion_data/smplx_322/{subset}" 
    test_text_path = f"inputs/motionx/motionx_seq_text_v1.1/{subset}"
    test_text, test_seq_names, test_motion = load_motionx_text(test_motion_path, test_text_path)
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
    print(f"Remove exist text in test_text: {len(exist_train_text)} on subset {subset}")

    test_seq_names = [seq_name for seq_name, text in zip(test_seq_names, test_text) if text is not None]
    test_motion = [m for m, text in zip(test_motion, test_text) if text is not None]
    test_text = [text for text in test_text if text is not None]
    return test_seq_names, test_text, test_motion


smplx_model = make_smplx("rich-smplx", gender="neutral")
hml3d_test_text = load_hml3d_text()
i = 0
all_seq_names = []
all_text = []
is_train = False
# is_train = True
is_simi = False
# is_simi = True
for subset in subsets:
    seq_names, text, all_motion = select_subset(subset)
    np.random.seed(i + 20)
    selected_index = np.random.permutation(len(seq_names))
    if subset == "idea400":
        N = 400
        # N = 150
        # N = 200
    else:
        N = 150
        N = 200
        # N = 100 

    if is_simi:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        test_text = [text[j] for j in selected_index]

        idea400_emb = model.encode(test_text)

        kmeans = KMeans(n_clusters=N)
        kmeans.fit(idea400_emb)
        cluster_center = kmeans.cluster_centers_
        closest_points = []
        closest_indices = []
        for i in range(N):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_points = idea400_emb[cluster_indices]
            distances = np.linalg.norm(cluster_points - cluster_center[i], axis=-1)
            closest_index = np.argmin(distances)
            closest_indices.append(cluster_indices[closest_index])

        # hml3d_emb = model.encode(hml3d_test_text)
        # similarities = model.similarity(idea400_emb, hml3d_emb).numpy()
        # max_cosine_similarity = np.max(similarities, axis=-1)
        # selected_index = np.argsort(max_cosine_similarity)
        # if not is_train:
        #     sit_num = 0
        #     stand_num = 0
        #     fitler_selected_index = []
        #     for ind in selected_index:
        #         if "sit" in text[ind]:
        #             sit_num += 1
        #             if sit_num > 50:
        #                 continue
        #         if "stand" in text[ind]:
        #             stand_num += 1
        #             if stand_num > 50:
        #                 continue
        #         fitler_selected_index.append(ind)
        #     selected_index = fitler_selected_index
    else:
        if is_train:
            selected_index = selected_index[N:]
        else:
            selected_index = selected_index[:N]

    
    i += 1
    # for j in selected_index:
        # wis3d = make_wis3d(output_dir="outputs/wis3d_idea400_gt", name=f"{j:03}")
        # motion = all_motion[j]
        # motion = torch.tensor(motion, dtype=torch.float32)
        # smpl_params = {"global_orient": motion[:, :3],
        #           "body_pose": motion[:, 3:66],
        #           "transl": motion[:, 309:312],
        #           "betas": None,
        #           }

        # joints = smpl_fk(smplx_model, **smpl_params)  # (F, 22, 3)
        # add_motion_as_lines(joints, wis3d, name=f"{text[j]}")

    for j in selected_index:
        seq_name_new = f"{subset}/{seq_names[j]}"
        all_seq_names.append(seq_name_new)
        all_text.append(text[j])
sit_num = 0
stand_num = 0
for t in all_text:
    if "sit" in t:
        sit_num += 1
    if "stand" in t:
        stand_num += 1
    print(t)

print(f"Select {len(all_seq_names)} sequences, sit {sit_num}, stand {stand_num}")

# save test_seq names
if is_simi:
    train_seq_names = []
    test_seq_names = []
    for j in selected_index:
        seq_name_new = f"{subset}/{seq_names[j]}"
        if j in closest_indices:
            test_seq_names.append(seq_name_new)
        else:
            train_seq_names.append(seq_name_new)

    save_file = "inputs/motionx/motionx_seq_text_v1.1/idea400_train_seq_names.json"
    with open(save_file, "w") as f:
        json.dump(train_seq_names, f, indent=4)

    save_file = "inputs/motionx/motionx_seq_text_v1.1/idea400_test_seq_names.json"
    with open(save_file, "w") as f:
        json.dump(test_seq_names, f, indent=4)

else:
    if is_train:
        save_file = "inputs/motionx/motionx_seq_text_v1.1/mixed_train_seq_names.json"
    else:
        save_file = "inputs/motionx/motionx_seq_text_v1.1/mixed_test_seq_names.json"
    
    with open(save_file, "w") as f:
        json.dump(all_seq_names, f, indent=4)


import ipdb;ipdb.set_trace()

