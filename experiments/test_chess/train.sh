#!/bin/bash
export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"
export PROJECT_ROOT=${PWD}
source ${PWD}/secrets.sh

source ~/miniconda3/bin/activate
conda activate JaxSeq2

# 4/18/2023

CUDA_VISIBLE_DEVICES=3,4 python -m llm_rl_scripts.chess.train_bc_gpt2 \
    HF \
    gpt2-xl \
    gcs://charlie-bucket2/datasets/chess_data/chess_data/train.jsonl \
    gcs://charlie-bucket2/datasets/chess_data/chess_data/val.jsonl \
    --exp-name "gpt2_xl_test1" \
    --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/chess_bc/ \
    --use-wandb \
    --wandb-project "chess_bc" \
    \
    --epochs 1 \
    --train-bsize 32 \
    --grad-accum-steps 1 \
    --eval-loss-bsize 32 \
    --eval-loss-batches 256 \
    --generation-bsize 4 \
    --generation-batches 2048 \
    --max-input-length 128 \
    --max-output-length 16 \
    \
    --log-every 256 \
    --eval-every-steps 1024 \
    --save-every-steps 1024 \
    --save-at-end \
    --no-save-train-state \
    \
    --data-mesh-shape -1 \
    --model-mesh-shape 1 \
    --fsdp \
    --gradient-checkpoint \

# CUDA_VISIBLE_DEVICES=3,4 python -m llm_rl_scripts.chess.train_bc_gpt2 \
#     HF \
#     gpt2 \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/train.jsonl \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/val.jsonl \
#     --exp-name "gpt2_small_test1" \
#     --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/chess_bc/ \
#     --use-wandb \
#     --wandb-project "chess_bc" \
#     \
#     --epochs 1 \
#     --train-bsize 32 \
#     --grad-accum-steps 1 \
#     --eval-loss-bsize 32 \
#     --eval-loss-batches 256 \
#     --generation-bsize 4 \
#     --generation-batches 2048 \
#     --max-input-length 128 \
#     --max-output-length 16 \
#     \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --save-every-steps 1024 \
#     --save-at-end \
#     --no-save-train-state \
#     \
#     --data-mesh-shape -1 \
#     --model-mesh-shape 1 \
#     --fsdp \
#     --gradient-checkpoint \
