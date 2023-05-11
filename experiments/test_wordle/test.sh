#!/bin/bash
export PYTHONPATH=${PWD}
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"
# export GCLOUD_PROJECT="rail-tpus"
# export GCLOUD_TOKEN_PATH=${HOME}/.config/gcloud/rail-tpus.json
export PROJECT_ROOT=${PWD}
source ${PWD}/secrets.sh

source ~/miniconda3/bin/activate
conda activate LLM_RL

# 5/6/2023

export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

python -m llm_rl_scripts.wordle.data_gen \
    --n-data 1000000 \
    --n-proc 96 \
    --vocab-path llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
    --prob-smart 0.33 \
    --out-path gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl
