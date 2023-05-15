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

# 4/13/2023

export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

CUDA_VISIBLE_DEVICES=5 python -m llm_rl_scripts.wordle.train_ppo \
    PARAMS \
    gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
    gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
    llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
    --exp-name None \
    --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_ppo_test1/ \
    --use-wandb \
    --wandb-project "LLM_RL_wordle_ppo" \
    --n-rollouts 512 \
    --train-bsize 16 \
    --grad-accum-steps 2 \
    --rollout-bsize 64 \
    --ppo-data-bsize 64 \
    --n-rounds 1000 \
    --epochs 4 \
    --log-every 32 \
    --weight-decay 1e-6 \
    --lr 3e-5 \
    --init-kl-coef 0.001 \
    --kl-target 0.1 \
    --kl-horizon 10000 \
    --value-loss-coef 1.0 \
    \
    --data-mesh-shape -1 \
    --fsdp-mesh-shape 1 \
    --model-mesh-shape 1 \
    \
    --bf16-activations \
    --no-save-best \
    --bc-loss-weight 1.0



# 5/11/2023

# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

# python -m llm_rl_scripts.wordle.train_bc \
#     CONFIG \
#     experiments/test_wordle/base_config.json \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name "wordle_gptj_config_test2" \
#     --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/ \
#     --use-wandb \
#     --wandb-project "wordle_bc" \
#     \
#     --epochs 100 \
#     --train-bsize 128 \
#     --grad-accum-steps None \
#     --max-length 128 \
#     --eval-loss-bsize 128 \
#     --eval-loss-batches 256 \
#     --policy-bsize 128 \
#     --policy-n-rollouts 256 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16 \
#     \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --save-every-steps 1024 \
#     --save-at-end \
#     --no-save-best \
#     --no-save-train-state \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \

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
