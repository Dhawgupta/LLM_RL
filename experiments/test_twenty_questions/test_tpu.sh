#!/bin/bash
export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"
export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"
export PROJECT_ROOT=${PWD}
source ${PWD}/secrets.sh

source ~/miniconda3/bin/activate
conda activate LLM_RL

# source ~/tpu_launcher2.sh /Users/charliesnell/current_projects/LLM_RL "~/LLM_RL"
# source ~/tpu_launcher2.sh /home/csnell/LLM_RL "~/LLM_RL"

# 4/22/2023

# tpu nlp-pod cl charlie-pod2 "source experiments/test_ppo/test_tpu.sh"

CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gptj.gptj_train \
    CONFIG \
    experiments/test_ppo/base_config.json \
    gcs://rail-tpus-csnell-us/test_ppo/10bit_data_multistep2.jsonl \
    gcs://rail-tpus-csnell-us/test_ppo/10bit_data_multistep2.jsonl \
    --epochs 1 \
    --train_bsize 64 \
    --eval-loss-bsize 64 \
    --log-every 256 \
    --eval-every-steps 1024 \
    --eval_loss_batches 64 \
    --generation_batches 64 \
    --lr 3e-4 \
    --outputs-path gcs://rail-tpus-csnell-us/LLM_RL_outputs/ppo_test_multistep2/ \
    --use-wandb \
    --wandb-project "ppo_test_multistep_initial_finetune" \
    --model-mesh-shape 1 \
    --data-mesh-shape 4 \
