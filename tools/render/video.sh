#!/bin/bash

# Blender 路径
BLENDER="/home/huaijin/install/blender-2.93.18-linux-x64/blender"

# 渲染脚本路径
RENDER_SCRIPT="tools/render/render_args.py"

# 定义参数集合，包含 MODE, MODEL, SUBSET, PATH, CAMERA_X, CAMERA_Y, CAMERA_ROT_X, CAMERA_ROT_Y, CAMERA_ROT_Z
CONFIGS=(
    # teaser
    "video singlemodel teaser ./outputs/dumped_teaser_singlemodel_mesh/0019_mesh.npy -7.1 -3.6 61 0.0 -68"
    "video singlemodel teaser ./outputs/dumped_teaser_singlemodel_mesh/0036_mesh.npy -2.5 11.4 64 0.0 -158"
    "video singlemodel teaser ./outputs/dumped_teaser_singlemodel_mesh/0007_mesh.npy -0.7 -11 64 0.0 -11"
    
    # comp_1
    "video singlemodel comp_basketball ./outputs/dumped_comp_basketball_singlemodel_mesh/0005_mesh.npy -7.1 -3.6 61 0.0 -68"
    "video mdm comp_basketball ./outputs/dumped_comp_basketball_mdm_mesh/0019_mesh.npy 4.9 5.6 56 0.0 140"
    "video motionclip comp_basketball ./outputs/dumped_comp_basketball_motionclip_mesh/0000_mesh.npy -6.7 0 54 0.0 270"
    "video mld comp_basketball ./outputs/dumped_comp_basketball_mld_mesh/0003_mesh.npy 7.0 0 54 0.0 90"

    # comp_2
    "video single t2m ./outputs/dumped_t2m_single_mesh/0926_mesh.npy -2.5 -8.0 60 0.0 -20"
    "video mdm comp_2 ./outputs/dumped_comp_2_mdm_mesh/0000_mesh.npy 7.0 0 54 0.0 90"
    "video mld comp_2 ./outputs/dumped_comp_2_mld_mesh/0002_mesh.npy 7.0 0 54 0.0 90"
    "video motionclip comp_2 ./outputs/dumped_comp_2_motionclip_mesh/0000_mesh.npy -6.7 0 54 0.0 270"

    # comp_3
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0001_mesh.npy -7.5 0 57 0.0 270"
    "video mld userstudy ./outputs/dumped_userstudy_mld_mesh/0001_mesh.npy 7.0 0 54 0.0 90"
    "video mdm userstudy ./outputs/dumped_userstudy_mdm_mesh/0001_mesh.npy 4.9 5.6 56 0.0 140"
    "video motionclip userstudy ./outputs/dumped_userstudy_motionclip_mesh/0001_mesh.npy -6.7 0 54 0.0 270"

    # 2d strategy_1
    "video single_fid308 t2m ./outputs/dumped_t2m_single_fid308_mesh/0003_mesh.npy -2.4 -8.5 60 0.0 -22"
    "video ours t2m ./outputs/dumped_t2m_ours_mesh/0003_mesh.npy 7.3 6.5 62 0.0 134"
    "video mas t2m ./outputs/dumped_t2m_mas_mesh/0003_mesh.npy 7.0 0 54 0.0 90"
    "video motionbert t2m ./outputs/dumped_t2m_motionbert_mesh/0003_mesh.npy 7.3 6.5 62 0.0 134"

    # 2d strategy_2
    "video single_fid308 t2m ./outputs/dumped_t2m_single_fid308_mesh/0211_mesh.npy 7.4 -7.3 65 0.0 46.7"
    "video ours t2m ./outputs/dumped_t2m_ours_mesh/0211_mesh.npy 7.4 -7.3 65 0.0 46.7"
    "video mas t2m ./outputs/dumped_t2m_mas_mesh/0211_mesh.npy 7.0 0 54 0.0 90"
    "video motionbert t2m ./outputs/dumped_t2m_motionbert_mesh/0211_mesh.npy 7.3 6.5 62 0.0 134"

    # ablation_1
    "video single_fid308 t2m ./outputs/dumped_t2m_single_fid308_mesh/3926_mesh.npy -2.5 -8.0 60 0.0 -20"
    "video single_cb t2m ./outputs/dumped_t2m_single_cb_mesh/3926_mesh.npy 7.0 0 54 0.0 90"
    "video single_scratch t2m ./outputs/dumped_t2m_single_scratch_mesh/3926_mesh.npy 7.0 0 54 0.0 90"
    "video single_3view t2m ./outputs/dumped_t2m_single_3view_mesh/3926_mesh.npy 7.0 0 54 0.0 90"
    "video single_5view t2m ./outputs/dumped_t2m_single_5view_mesh/3926_mesh.npy 7.0 0 54 0.0 90"

    # ablation_2
    "video single_fid308 t2m ./outputs/dumped_t2m_single_fid308_mesh/3212_mesh.npy -2.5 -8.0 60 0.0 -20"
    "video single_cb t2m ./outputs/dumped_t2m_single_cb_mesh/3212_mesh.npy 7.0 0 54 0.0 90"
    "video single_scratch t2m ./outputs/dumped_t2m_single_scratch_mesh/3212_mesh.npy 7.0 0 54 0.0 90"
    "video single_3view t2m ./outputs/dumped_t2m_single_3view_mesh/3212_mesh.npy 7.0 0 54 0.0 90"
    "video single_5view t2m ./outputs/dumped_t2m_single_5view_mesh/3212_mesh.npy 7.0 0 54 0.0 90"
)

# 遍历每个配置并执行渲染
for CONFIG in "${CONFIGS[@]}"; do
    # 分割配置项
    IFS=" " read -r MODE MODEL SUBSET PATH CAMERA_X CAMERA_Y CAMERA_ROT_X CAMERA_ROT_Y CAMERA_ROT_Z <<< "$CONFIG"

    echo "正在渲染: MODE=$MODE, MODEL=$MODEL, SUBSET=$SUBSET, PATH=$PATH, CAMERA_XY=[$CAMERA_X, $CAMERA_Y], CAMERA_ROT=[$CAMERA_ROT_X, $CAMERA_ROT_Y, $CAMERA_ROT_Z]"

    # 执行渲染命令
    $BLENDER --background --python $RENDER_SCRIPT -- \
        --mode "$MODE" \
        --path "$PATH" \
        --camera-xy "$CAMERA_X" "$CAMERA_Y" \
        --camera-rot "$CAMERA_ROT_X" "$CAMERA_ROT_Y" "$CAMERA_ROT_Z"
done

echo "所有渲染任务完成！"