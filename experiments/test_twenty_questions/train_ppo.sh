#!/bin/bash

export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_DISABLE_JIT=0
# export CUDA_VISIBLE_DEVICES=""
# export TOKENIZERS_PARALLELISM=false
export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"
export PROJECT_ROOT=${PWD}

# python -m llm_rl_scripts.twenty_questions.train_ppo \
#     PARAMS \
#     gcs://rail-tpus-charles-2/ILQL5/outputs/twenty_questions/bc_gpt2med_test8_converted/model \
#     --exp-name ppo_gpt2med_test1 \
#     --outputs-path gcs://rail-tpus-charles-2/LLM_RL/outputs/twenty_question/ \
#     --train-bsize 32 \
#     --n-rounds 1000 \
#     --epochs 10 \
#     --n-rollouts 158 \
#     --rollout-bsize 1 \
#     --log-every 4 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --wandb-project twenty_questions-ppo \
#     --save-every-rounds 50 \
#     --init-kl-coef 0.001 \
#     --kl-target 0.1 \
#     --kl-horizon 10000 \
#     --env-deterministic \
#     --oracle-model-path gcs://rail-tpus-charles-2/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1_converted/model \
#     # --use-wandb \


python -m llm_rl_scripts.twenty_questions.train_ppo \
    HF \
    gpt2-medium \
    --exp-name ppo_gpt2med_test1 \
    --outputs-path gcs://rail-tpus-charles-2/LLM_RL/outputs/twenty_question/ \
    --train-bsize 4 \
    --n-rounds 1000 \
    --epochs 10 \
    --n-rollouts 8 \
    --rollout-bsize 1 \
    --log-every 4 \
    --weight-decay 1e-6 \
    --lr 3e-5 \
    --wandb-project twenty_questions-ppo \
    --save-every-rounds 50 \
    --init-kl-coef 0.001 \
    --kl-target 0.1 \
    --kl-horizon 10000 \
    --env-deterministic \
    --oracle-model-path gcs://rail-tpus-charles-2/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1_converted/model \

