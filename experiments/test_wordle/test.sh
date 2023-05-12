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

# 5/11/2023

export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

python -m llm_rl_scripts.wordle.train_bc \
    CONFIG \
    experiments/test_wordle/base_config.json \
    gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
    gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
    llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
    --exp-name "wordle_gptj_config_test2" \
    --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/ \
    --use-wandb \
    --wandb-project "wordle_bc" \
    \
    --epochs 100 \
    --train-bsize 128 \
    --grad-accum-steps None \
    --max-length 128 \
    --eval-loss-bsize 128 \
    --eval-loss-batches 256 \
    --policy-bsize 128 \
    --policy-n-rollouts 256 \
    --policy-max-input-length 128 \
    --policy-max-output-length 16 \
    \
    --log-every 256 \
    --eval-every-steps 1024 \
    --save-every-steps 1024 \
    --save-at-end \
    --no-save-best \
    --no-save-train-state \
    \
    --data-mesh-shape -1 \
    --fsdp-mesh-shape 1 \
    --model-mesh-shape 1 \

# python -m llm_rl_scripts.wordle.data_gen \
#     --n-data 1000000 \
#     --n-proc 96 \
#     --vocab-path llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --prob-smart 0.33 \
#     --out-path gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl

# python -m llm_rl_scripts.wordle.data_gen \
#     --n-data 100000 \
#     --n-proc 96 \
#     --vocab-path llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --prob-smart 0.33 \
#     --out-path gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl

# 5/6/2023

# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

# python -m llm_rl_scripts.chess.train_bc_gpt2 \
#     CONFIG \
#     experiments/test_wordle/base_config.json \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name "wordle_gptj_config_test1" \
#     --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/ \
#     --use-wandb \
#     --wandb-project "wordle_bc" \
#     \
#     --epochs 1 \
#     --train-bsize 64 \
#     --grad-accum-steps 1 \
#     --max-length 256 \
#     --eval-loss-bsize 64 \
#     --eval-loss-batches 256 \
#     --policy-bsize 64 \
#     --policy-n-rollouts 256 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 128 \
#     \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --save-every-steps 1024 \
#     --save-at-end \
#     --no-save-best \
#     --no-save-train-state \
#     \
#     --data-mesh-shape -1 \
#     --model-mesh-shape 1 \
#     # --fsdp \
#     # --gradient-checkpoint \

# python -m llm_rl_scripts.wordle.data_gen \
#     --n-data 1000000 \
#     --n-proc 96 \
#     --vocab-path llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --prob-smart 0.33 \
#     --out-path gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl
