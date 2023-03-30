#!/bin/bash

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="/home/csnell/.config/gcloud/civic-boulder-204700-V2.json"
export WANDB_API_KEY="6342624bbf30593fb0350a9198db14934332a574"
export PROJECT_ROOT="$HOME/LLM_RL/"
export PYTHONPATH=${HOME}/LLM_RL/src/:${HOME}/LLM_RL/scripts/:${HOME}/LLM_RL/JaxSeq2/src/:${HOME}/LLM_RL/JaxSeq2/examples/

# python -m PPO_scripts.generate_test_data \
#     --n 10 \
#     --output_path data/test_ppo/10bit_data.jsonl \

# python -m gptj_scripts.gptj_train \
#     CONFIG \
#     experiments/test_ppo/base_config.json \
#     data/test_ppo/10bit_data.jsonl \
#     data/test_ppo/10bit_data.jsonl \
#     --epochs 1000 \
#     --train_bsize 64 \
#     --eval-loss-bsize 64 \
#     --log-every 256 \
#     --eval-every 1024 \
#     --lr 3e-4 \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/ \
#     --use-wandb \
#     --wandb-project "ppo_test_initial_finetune" \

# python -m misc_scripts.export_checkpoint \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \

# python -m gptj_scripts.gptj_serve \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
#     --host 0.0.0.0 \
#     --port 8000 \

python -m PPO_scripts.ppo_test \
    PARAMS \
    gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
    data/test_ppo/10bit_data.jsonl \
    data/test_ppo/10bit_data.jsonl \
