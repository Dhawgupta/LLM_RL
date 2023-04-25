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

# 4/23/2023

export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH=${HOME}/.config/gcloud/rail-tpus.json

# python -m llm_rl_scripts.ppo.generate_test_data_multichain \
#     --n 10 \
#     --output_path data/test_ppo/10bit_data_multichain.jsonl \

# python -m examples_jaxseq.misc.export_checkpoint \
#     gcs://rail-tpus-csnell-us/LLM_RL_outputs/ppo_test_multistep2/exp.2023-04-22-21-52-41.254.015006f2e15811ed89e155792b4c6f0d/best/ \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gptj.gptj_serve \
#     PARAMS \
#     gcs://rail-tpus-csnell-us/LLM_RL_outputs/ppo_test_multistep2/exp.2023-04-22-21-52-41.254.015006f2e15811ed89e155792b4c6f0d/best/ \
#     --host 0.0.0.0 \
#     --port 8099 \

CUDA_VISIBLE_DEVICES=5 python -m llm_rl_scripts.ppo.ppo_test_multistep \
    PARAMS \
    gcs://rail-tpus-csnell-us/LLM_RL_outputs/ppo_test_multistep2/exp.2023-04-22-21-52-41.254.015006f2e15811ed89e155792b4c6f0d/best \
    --exp-name None \
    --outputs-path gcs://rail-tpus-csnell-us/LLM_RL_outputs/ppo_test_multistep/ \
    --train-bsize 32 \
    --n-rounds 100 \
    --epochs 4 \
    --log-every 4 \
    --weight-decay 1e-6 \
    --lr 1e-5 \
    --use-wandb \
    --wandb-project "rlhf_multistep_binary_test" \
    --save-every-rounds 1 \
    --init-kl-coef 0.001 \
    # --kl-target 0.1 \
    # --kl-horizon 10000 \

# 4/22/2023

# python -m examples_jaxseq.gptj.gptj_serve \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep/exp.2023-04-22-01-11-54.651.ab9b9eb8e0aa11ed8df5000000acfe80/best \
#     --host 0.0.0.0 \
#     --port 8099 \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gptj.gptj_train \
#     CONFIG \
#     experiments/test_ppo/base_config.json \
#     data/test_ppo/10bit_data_multistep2.jsonl \
#     data/test_ppo/10bit_data_multistep2.jsonl \
#     --epochs 1 \
#     --train_bsize 64 \
#     --eval-loss-bsize 64 \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --eval-batches 64 \
#     --lr 3e-4 \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep2/ \
#     --use-wandb \
#     --wandb-project "ppo_test_multistep_initial_finetune" \

# python -m llm_rl_scripts.ppo.generate_test_data_multistep \
#     --n 10 \
#     --output_path data/test_ppo/10bit_data_multistep2.jsonl \

# python -m examples_jaxseq.gptj.gptj_serve \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep/exp.2023-04-22-01-11-54.651.ab9b9eb8e0aa11ed8df5000000acfe80/best \
#     --host 0.0.0.0 \
#     --port 8099 \

# python -m examples_jaxseq.misc.export_checkpoint \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep/exp.2023-04-22-01-11-54.651.ab9b9eb8e0aa11ed8df5000000acfe80/best \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gptj.gptj_train \
#     CONFIG \
#     experiments/test_ppo/base_config.json \
#     data/test_ppo/10bit_data_multistep.jsonl \
#     data/test_ppo/10bit_data_multistep.jsonl \
#     --epochs 1 \
#     --train_bsize 64 \
#     --eval-loss-bsize 64 \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --lr 3e-4 \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep/ \
#     --use-wandb \
#     --wandb-project "ppo_test_multistep_initial_finetune" \

# 4/21/2023

# python -m llm_rl_scripts.ppo.generate_test_data_multistep \
#     --n 10 \
#     --output_path data/test_ppo/10bit_data_multistep.jsonl \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gptj.gptj_train \
#     CONFIG \
#     experiments/test_ppo/base_config.json \
#     data/test_ppo/10bit_data_multistep.jsonl \
#     data/test_ppo/10bit_data_multistep.jsonl \
#     --epochs 1 \
#     --train_bsize 64 \
#     --eval-loss-bsize 64 \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --lr 3e-4 \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_multistep/ \
#     --use-wandb \
#     --wandb-project "ppo_test_multistep_initial_finetune" \

# 4/17/2023

# CUDA_VISIBLE_DEVICES=4 python -m examples_jaxseq.gpt2.gpt2_train \
#     CONFIG \
#     experiments/test_ppo/gpt2_base_config.json \
#     data/test_ppo/10bit_data.jsonl \
#     data/test_ppo/10bit_data.jsonl \
#     --epochs 1000 \
#     --train-bsize 64 \
#     --eval-loss-bsize 64 \
#     --log-every 256 \
#     --eval-every-steps 1024 \
#     --lr 3e-4 \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test_gpt2/ \
#     --use-wandb \
#     --wandb-project "gpt2_ppo_test_initial_finetune" \

# CUDA_VISIBLE_DEVICES=4 python -m llm_rl_scripts.ppo.ppo_test_gpt2 \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/ \
#     --train-bsize 32 \
#     --n-rounds 100 \
#     --epochs 4 \
#     --log-every 4 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --use-wandb \
#     --wandb-project "rlhf_binary_test" \
#     --save-every-rounds 1 \
#     --init-kl-coef 0.001 \
#     --kl-target 0.1 \
#     --kl-horizon 10000 \

# CUDA_VISIBLE_DEVICES=3 python -m llm_rl_scripts.ppo.ppo_test \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/ \
#     --train-bsize 32 \
#     --n-rounds 100 \
#     --epochs 4 \
#     --log-every 4 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --use-wandb \
#     --wandb-project "rlhf_binary_test" \
#     --save-every-rounds 1 \
#     --init-kl-coef 0.001 \
#     --kl-target 0.1 \
#     --kl-horizon 10000 \

# 4/12/2023

# CUDA_VISIBLE_DEVICES=3 python -m llm_rl_scripts.ppo.ppo_test \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
#     --exp-name None \
#     --outputs-path gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/ \
#     --train-bsize 32 \
#     --n-rounds 100 \
#     --epochs 4 \
#     --log-every 4 \
#     --weight-decay 1e-6 \
#     --lr 3e-5 \
#     --use-wandb \
#     --wandb-project "rlhf_binary_test" \
#     --save-every-rounds 1 \
#     --init-kl-coef 0.001 \
#     --kl-target None \
#     --horizon None \

# older

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

# CUDA_VISIBLE_DEVICES=3 python -m llm_rl_scripts.ppo.ppo_test \
#     PARAMS \
#     gcs://charlie-bucket2/LLM_RL_outputs/ppo_test/exp.2023-03-25-02-33-00.401/best \
#     data/test_ppo/10bit_data.jsonl \
#     data/test_ppo/10bit_data.jsonl \
#     --train-bsize 1 \
