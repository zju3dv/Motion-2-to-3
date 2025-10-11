#!/bin/bash

# Blender 路径
BLENDER="/home/huaijin/install/blender-2.93.18-linux-x64/blender"

# 渲染脚本路径
RENDER_SCRIPT="tools/render/render_args.py"

# 定义参数集合，包含 MODE, MODEL, SUBSET, PATH, CAMERA_X, CAMERA_Y, CAMERA_ROT_X, CAMERA_ROT_Y, CAMERA_ROT_Z
CONFIGS=(
    # user study
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0002_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0003_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0004_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0005_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0006_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0007_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0008_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0009_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0010_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0011_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0012_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0013_mesh.npy -7.5 0 57 0.0 270"
    "video singlemodel userstudy ./outputs/dumped_userstudy_singlemodel_mesh/0014_mesh.npy -7.5 0 57 0.0 270"

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