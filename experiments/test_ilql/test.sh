#!/bin/bash
export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"
# export GCLOUD_PROJECT="rail-tpus"
# export GCLOUD_TOKEN_PATH=${HOME}/.config/gcloud/rail-tpus.json
export PROJECT_ROOT=${PWD}
source ${PWD}/secrets.sh

source ~/miniconda3/bin/activate
conda activate LLM_RL

# 4/24/2023

CUDA_VISIBLE_DEVICES=5 python -m llm_rl_scripts.ilql.ilql_test \
    PARAMS \
    gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
    gcs://charlie-bucket2/datasets/test_ppo/10bit_data.jsonl \
    --exp-name None \
    --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ilql_test/ \
    --train-bsize 32 \
    --epochs 4 \
    --log-every 4 \
    --weight-decay 1e-6 \
    --lr 3e-5 \
    --use-wandb \
    --wandb-project "ilql_binary_test" \

