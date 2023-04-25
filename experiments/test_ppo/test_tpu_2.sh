#!/bin/bash

export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_JIT=1
# export CUDA_VISIBLE_DEVICES=""
# export TOKENIZERS_PARALLELISM=false
export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"
export PROJECT_ROOT=${PWD}


python -m llm_rl_scripts.ppo.ppo_test_gpt2 \
    HF \
    gpt2-medium \
    --exp-name test \
    --outputs-path ./outputs/ppo_test/ \
    --train-bsize 4 \
    --n-rounds 100 \
    --n-rollouts 8 \
    --rollout-bsize 4 \
    --epochs 4 \
    --log-every 4 \
    --weight-decay 1e-6 \
    --lr 3e-5 \
    --save-every-rounds 1 \
    --init-kl-coef 0.001 \
    --kl-target 0.1 \
    --kl-horizon 10000 \

# python -m llm_rl_scripts.ppo.ppo_test \
#     HF \
#     EleutherAI/gpt-j-6B \
#     --exp-name test2 \
#     --outputs-path ./outputs/ppo_test/ \
#     --train-bsize 1 \
#     --n-rounds 100 \
#     --n-rollouts 8 \
#     --rollout-bsize 1 \
#     --epochs 4 \
#     --log-every 4 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --save-every-rounds 1 \
#     --init-kl-coef 0.001 \
#     --kl-target 0.1 \
#     --kl-horizon 10000 \
