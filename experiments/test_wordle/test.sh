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

# 9/20/23

source ~/miniconda3/bin/activate
conda activate LLM_RL


export GCLOUD_PROJECT="civic-boulder-204700"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

python -m llm_rl_scripts.wordle.train_bc_gpt2 \
    HF \
    gpt2 \
    gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
    gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
    llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
    --exp-name "wordle_gpt2_test2" \
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
    --data-mesh-shape 1 \
    --fsdp-mesh-shape -1 \
    --model-mesh-shape 1 \


# source ~/miniconda3/bin/activate
# conda activate LLM_RL

# 9/17/23

# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

# # bc

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/eval_bc_sample/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/eval_bc_greedy/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16 \
#     --no-policy-do-sample




# export GCLOUD_PROJECT="rail-tpus"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"

# 50% bc

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_50/wordle_gptj_config_test2_filtered_50.2023-09-16-06-18-04.344.cb859284545811ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_50/wordle_gptj_config_test2_filtered_50.2023-09-16-06-18-04.344.cb859284545811ee9b91dd1a943a02f5/last/eval_bc_sample/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_50/wordle_gptj_config_test2_filtered_50.2023-09-16-06-18-04.344.cb859284545811ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_50/wordle_gptj_config_test2_filtered_50.2023-09-16-06-18-04.344.cb859284545811ee9b91dd1a943a02f5/last/eval_bc_greedy/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16 \
#     --no-policy-do-sample

# 30% bc

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_30/wordle_gptj_config_test2_filtered_30.2023-09-16-11-36-39.799.4d388440548511ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_30/wordle_gptj_config_test2_filtered_30.2023-09-16-11-36-39.799.4d388440548511ee9b91dd1a943a02f5/last/eval_bc_sample/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_30/wordle_gptj_config_test2_filtered_30.2023-09-16-11-36-39.799.4d388440548511ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_30/wordle_gptj_config_test2_filtered_30.2023-09-16-11-36-39.799.4d388440548511ee9b91dd1a943a02f5/last/eval_bc_greedy/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16 \
#     --no-policy-do-sample

# 10% bc

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_10/wordle_gptj_config_test2_filtered_10.2023-09-16-14-43-23.058.62e230a6549f11ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_10/wordle_gptj_config_test2_filtered_10.2023-09-16-14-43-23.058.62e230a6549f11ee9b91dd1a943a02f5/last/eval_bc_sample/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16

# python -m llm_rl_scripts.wordle.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_10/wordle_gptj_config_test2_filtered_10.2023-09-16-14-43-23.058.62e230a6549f11ee9b91dd1a943a02f5/last/ \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_10/wordle_gptj_config_test2_filtered_10.2023-09-16-14-43-23.058.62e230a6549f11ee9b91dd1a943a02f5/last/eval_bc_greedy/ \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --policy-n-rollouts 4096 \
#     --policy-bsize 128 \
#     --policy-max-input-length 128 \
#     --policy-max-output-length 16 \
#     --no-policy-do-sample


# 9/15/23

# export GCLOUD_PROJECT="rail-tpus"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"

# python -m llm_rl_scripts.wordle.train_bc \
#     CONFIG \
#     experiments/test_wordle/base_config.json \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data1_filtered_50.0.jsonl \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name "wordle_gptj_config_test2_filtered_50" \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_50/ \
#     --use-wandb \
#     --wandb-project "wordle_bc" \
#     \
#     --epochs 20 \
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

# python -m llm_rl_scripts.wordle.train_bc \
#     CONFIG \
#     experiments/test_wordle/base_config.json \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data1_filtered_30.0.jsonl \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name "wordle_gptj_config_test2_filtered_30" \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_30/ \
#     --use-wandb \
#     --wandb-project "wordle_bc" \
#     \
#     --epochs 20 \
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

# python -m llm_rl_scripts.wordle.train_bc \
#     CONFIG \
#     experiments/test_wordle/base_config.json \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data1_filtered_10.0.jsonl \
#     gcs://rail-tpus-csnell-us/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name "wordle_gptj_config_test2_filtered_10" \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/wordle_filtered_bc_10/ \
#     --use-wandb \
#     --wandb-project "wordle_bc" \
#     \
#     --epochs 20 \
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

# python -m llm_rl_scripts.wordle.optimal_perf

# 5/21/2023

# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

# python -m llm_rl_scripts.wordle.train_mc_returns \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_ilql_test1/ \
#     --use-wandb \
#     --wandb-project "LLM_RL_wordle_ilql" \
#     --train-bsize 8 \
#     --grad-accum-steps 4 \
#     --policy-bsize 64 \
#     --policy-n-rollouts 512 \
#     --eval-loss-bsize 8 \
#     --eval-loss-batches 64 \
#     --epochs 1000 \
#     --log-every 64 \
#     --eval-every-steps 1024 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --bf16-activations \
#     --no-save-best \
#     --beta 16.0 \
#     --gamma 1.0

# python -m llm_rl_scripts.wordle.train_ilql \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_ilql_test1/ \
#     --use-wandb \
#     --wandb-project "LLM_RL_wordle_ilql" \
#     --train-bsize 8 \
#     --grad-accum-steps 4 \
#     --policy-bsize 64 \
#     --policy-n-rollouts 512 \
#     --eval-loss-bsize 8 \
#     --eval-loss-batches 64 \
#     --epochs 1000 \
#     --log-every 64 \
#     --eval-every-steps 1024 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --bf16-activations \
#     --no-save-best \
#     --beta 16.0

# 4/13/2023

# export GCLOUD_PROJECT="civic-boulder-204700"
# export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/civic-boulder-204700-V2.json"

# CUDA_VISIBLE_DEVICES=5 python -m llm_rl_scripts.wordle.train_ilql \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data_eval1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_ilql_test1/ \
#     --use-wandb \
#     --wandb-project "LLM_RL_wordle_ilql" \
#     --train-bsize 8 \
#     --grad-accum-steps 4 \
#     --policy-bsize 64 \
#     --policy-n-rollouts 512 \
#     --eval-loss-bsize 8 \
#     --eval-loss-batches 64 \
#     --epochs 1000 \
#     --log-every 64 \
#     --eval-every-steps 1024 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --bf16-activations \
#     --no-save-best \
#     --beta 16.0

# CUDA_VISIBLE_DEVICES=5 python -m llm_rl_scripts.wordle.train_ppo \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gptj_config_test2.2023-05-12-17-01-25.893.a16043b2f0e611ed890c5d20da9db470/step_237568/ \
#     gcs://charlie-bucket2/LLM_RL_data/wordle/bc_data1.jsonl \
#     llm_rl_scripts/wordle/vocab/wordle_official_400.txt \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_ppo_test1/ \
#     --use-wandb \
#     --wandb-project "LLM_RL_wordle_ppo" \
#     --n-rollouts 512 \
#     --train-bsize 8 \
#     --grad-accum-steps 4 \
#     --rollout-bsize 64 \
#     --ppo-data-bsize 64 \
#     --n-rounds 1000 \
#     --epochs 4 \
#     --log-every 32 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --init-kl-coef 0.001 \
#     --kl-target 0.1 \
#     --kl-horizon 10000 \
#     --value-loss-coef 1.0 \
#     \
#     --data-mesh-shape -1 \
#     --fsdp-mesh-shape 1 \
#     --model-mesh-shape 1 \
#     \
#     --bf16-activations \
#     --no-save-best \
#     --bc-loss-weight 10.0



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
