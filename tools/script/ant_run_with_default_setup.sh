#!/bin/bash
# Usage: ./tools/script/ant_run_with_default_setup.sh <main_py> <a string of args>

# ===== Extra setup for container ===== #
CODE_ROOT=$(pwd)
pip install einops==0.6.1 decord
pip install -e .

# Very strange, it seems that Ant-Cloud may install a different version of lightning  
pip install pytorch-lightning==2.0.9 lightning-cloud==0.5.38
# pip install tensorboardX==2.6.2.2

pip list | grep lightning
pip list | grep tensorboard  # TODO: it seems version is not the same as my local env

# ===== Extra setup for project ===== #
CODE_INPUTS=${CODE_ROOT}/inputs
OSS_INPUTS="oss://antsys-vilab/shenzehong/proj_data/hmr4d/inputs"
NAS2_INPUTS=/input_ssd/shenzehong/hmr4d  # This very slow
NAS_OUTPUTS=/input/shenzehong/proj_data/hmr4d/outputs

# Set inputs/
mkdir -p ${CODE_INPUTS}
ln -s ${NAS2_INPUTS}/RICH ${CODE_INPUTS}/RICH
ln -s ${NAS2_INPUTS}/amass ${CODE_INPUTS}/amass
ln -s ${NAS2_INPUTS}/checkpoints ${CODE_INPUTS}/checkpoints

# Set outputs/
ln -s $NAS_OUTPUTS ./outputs

# ===== Launch job ===== #
# Set env vars
HYDRA_FULL_ERROR=1

# List all ddp-related env vars
echo "\
1. Check k8s master EnvVar: MASTER Address=${MASTER_ADDR}, Port=${MASTER_PORT}
2. Check td master EnvVar: TD_MASTER_IP=${TD_MASTER_IP}, TD_MASTER_PORT=${TD_MASTER_PORT}
3. Check common EnvVar: NODE_SIZE=${NODE_SIZE}, NODE_RANK=${NODE_RANK}
        (should equal to) WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}
4. TD_GPU_PER_NODE=${TD_GPU_PER_NODE}
"

PyFile=$1
# echo "Use torch.distributed.run to execute:" # TODO: fix potential bug
echo "python ${PyFile} ${@:2}"

HYDRA_FULL_ERROR=1 python ${PyFile} ${@:2}

# python -m torch.distributed.run \
#     --master_addr ${TD_MASTER_IP} \
#     --master_port ${TD_MASTER_PORT} \
#     --nnodes ${NODE_SIZE} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${TD_GPU_PER_NODE} \
#     ${PyFile} ${@:2}
