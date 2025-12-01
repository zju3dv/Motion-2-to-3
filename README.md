# Motion-2-to-3: Leveraging 2D Motion Data to Boost 3D Motion Generation
### [Project Page](https://zju3dv.github.io/Motion-2-to-3) | [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_Motion-2-to-3_Leveraging_2D_Motion_Data_for_3D_Motion_Generations_ICCV_2025_paper.pdf)
> Motion-2-to-3: Leveraging 2D Motion Data to Boost 3D Motion Generation  
> [Ruoxi Guo](https://www.researchgate.net/profile/Ruoxi-Guo-2)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Zehong Shen](https://zehongs.github.io/),
[Qing Shuai](https://chingswy.github.io/),
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Zhumei Wang](https://zju3dv.github.io/gvhmr),
[Yajiao Dong](https://zju3dv.github.io/gvhmr),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Taku Komura](https://i.cs.hku.hk/~taku),
[Sida Peng](https://pengsida.net/),
[Xiaowei Zhou](https://xzhou.me/)  
> ICCV 2025

<p align="center">
    <img src=docs/image/teaser_v3_c.png />
</p>

## Code Release
The Inference Code and Checkpoints are available now!

## Setup

### Installation

```bash
conda create -y -n hmr4d python=3.8
conda activate hmr4d
pip install -r requirements.txt
pip install -e .
# NOTE: if you want to editable install hmr4d in other repo,  try adding "python.analysis.extraPaths": ["path/to/your/package"] to your settings.json

### For rendering
conda create -y -n render2d3d python=3.10
conda activate render2d3d
pip install -r requirements_render.txt
pip install -e .
```

### Folder Structure

You should prepare the models and data before training and testing. And they should be placed in the following structure:
```bash
inputs/
├── checkpoints/ # models to use
│   ├── body_models/smplh/
│   ├── glove/
│   ├── huggingface/
│   └── t2m/
│       ├── length_est_bigru/
│       └── text_mot_match/
├── hml3d/ # 3d data
└── ... not needed now
```

#### 1. HumanML3D
You can download the HumanML3D training data here:
https://drive.google.com/drive/folders/1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu

Put the files under folder `inputs/hml3d`.
#### 2. Body Models
```bash
You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:
```bash
# Unzip from SMPL_python_v.1.1.0.zip and rename the pkl file
inputs/checkpoints/body_models/
└── smplh/
    └── SMPLH_{GENDER}.pkl  # SMPLH 
```
#### 3. CLIP
Put CLIP checkpoint here: `inputs/checkpoints/huggingface/clip-vit-base-patch32`.

If you do not have CLIP checkpoint locally, you can uncomment L33 in `hmr4d/network/clip.py` to use CLIP from OpenAI. Remember to set proxy if you are in China.

#### 4. GLOVE
```bash
bash tools/download_glove.sh
```
Put glove at `inputs/checkpoints/glove`.

#### 5. T2M
You can find the t2m file here:
https://huggingface.co/juracera/Motion-2-to-3/tree/main

Put the two folders under `inputs/checkpoints/t2m`.
## Test
### Checkpoints
You can download the checkpoints here:
https://huggingface.co/juracera/Motion-2-to-3/tree/main

Save the one checkpoints in this directory:
`outputs/HumanML3D_2dmotionmv_nr/mdm-hmlfinetune/cpurand_best.ckpt`

**Notice: The provided checkpoints is an updated version. The former one has different performance between machines. This version has a more uniform performance trained on different machines such as V100, RTX4090.**

**We recommend and welcome you to use this checkpoint for direct inference.**

The referenced metric on HumanML3D, evaluated on an RTX2070 is showed in the table below.

*Metrics like FID can only be use as references in motion generation. We are more focused on its ability to generalize compared to former methods.*

| Metric (Our / Ground-truth) | Value         |
| --------------------------- | ------------- |
| **Matching score**          | 3.517 / 3.000 |
| **Diversity**               | 9.573 / 9.154 |
| **R-precision top-1**       | 0.394 / 0.503 |
| **R-precision top-2**       | 0.586 / 0.698 |
| **R-precision top-3**       | 0.702 / 0.795 |
| **FID**                     | 0.427 / 0.002 |


### Test the MV model on HumanML3D
```bash
HYDRA_FULL_ERROR=1 python tools/train.py exp=mas/video_motion2dmv_nr/mdm_test global/task=motion2dmv/single_test2dmv_nr 
```
Use saved results for fast metric calculation.

Check `hmr4d/model/mas/callbacks/metric_generation.py`. 

This is not a runnable file, but you can see where the pth is saved.

### Test the MV model using New Text Prompts
```bash
HYDRA_FULL_ERROR=1 python tools/train.py exp=mas/video_motion2dmv_nr/mdm_test global/task=motion2dmv/single_newtext_nr 
```
You can use dumper to save the generation results:
```bash
# hmr4d/configs/global/task/motion2dmv/single_newtext_nr.yaml
# Callbacks
callbacks:
  dumper: # uncomment to enable generation result dumping
    _target_: hmr4d.utils.callbacks.dump_results.ResultDumper
    saved_dir: ./outputs/dumped_single_newtext_nr
```
## Visualization

### 2Dto3D (Wis3D)

Visualize results with Wis3D.

Add `+global/append=add_cb_Wis3d_2dto3dVisualizer`

You can change name in the callback to change saved directory.
```bash
hmr4d/configs/global/append/add_cb_Wis3d_2dto3dVisualizer.yaml
```
Then
```bash
cd outputs/wis3d
wis3d --vis_dir ./ --host localhost --port 7654
```

NOTE: Wis3d costs a lot cpu, if you do not use it, please stop it and remove the saved results.

### Fit SMPL
Please refer to 
```bash
python tools/mas/joints2smpl.py
```
You may need to adjust this a bit.
## Prepare data
### Convert HumanML3D to joints3d.pth

Convert HumanML3D data to 3d joint positions. 

Put required AMASS dataset under `./inputs/amass/smplhg_raw`

Please also download humanact12 from [HumanML3d](https://github.com/EricGuo5513/HumanML3D/tree/main/pose_data) and unzip it under `./inputs`, then you will get `./inputs/humanact12`

Put HumanML3D required `index.csv`, `texts`, `train.txt`, `train_val.txt`, `val.txt`, and `test.txt` under `./inputs/hml3d`.

Use the following script
```bash
python tools/mas/export_HumanML3D.py
```

This will save a `joints3d.pth` under `./inputs/hml3d`.

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@InProceedings{Guo_2025_ICCV,
    author    = {Guo, Ruoxi and Pi, Huaijin and Shen, Zehong and Shuai, Qing and Hu, Zechen and Wang, Zhumei and Dong, Yajiao and Hu, Ruizhen and Komura, Taku and Peng, Sida and Zhou, Xiaowei},
    title     = {Motion-2-to-3: Leveraging 2D Motion Data for 3D Motion Generations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {14305-14316}
}
```
