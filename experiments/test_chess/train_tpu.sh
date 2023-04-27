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

# source ~/tpu_launcher2.sh /Users/charliesnell/current_projects/LLM_RL "~/LLM_RL"
# source ~/tpu_launcher2.sh /home/csnell/LLM_RL "~/LLM_RL"

# 4/26/2023

# tpu nlp-pod cl charlie-pod "source experiments/test_chess/train_tpu.sh"

# python -m llm_rl_scripts.chess.train_bc_llama \
#     PARAMS \
#     gcs://llama_weights_charlie/7B/ \
#     gcs://llama_weights_charlie/tokenizer.model \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/train_2.jsonl \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/val_2.jsonl \
#     --exp-name "llama_7B_test2" \
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
#     --model-mesh-shape 4 \
#     --fsdp \
#     --gradient-checkpoint \

# 4/20/2023

# tpu nlp-pod cl charlie-pod "source experiments/test_chess/train_tpu.sh"

# python -m llm_rl_scripts.chess.train_bc_llama \
#     PARAMS \
#     gcs://llama-weights-charlie/llama_weights_charlie/30B/ \
#     gcs://llama-weights-charlie/llama_weights_charlie/tokenizer.model \
#     gcs://rail-tpus-csnell-us/datasets/chess_data/chess_data/train_2.jsonl \
#     gcs://rail-tpus-csnell-us/datasets/chess_data/chess_data/val_2.jsonl \
#     --exp-name "llama_30B_test1" \
#     --outputs-path gcs://rail-tpus-csnell-us/JaxSeq2_outputs/chess_bc/ \
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
#     --model-mesh-shape 4 \
#     --fsdp \
#     --gradient-checkpoint \


# python -m llm_rl_scripts.chess.train_bc_llama \
#     PARAMS \
#     gcs://llama_weights_charlie/13B/ \
#     gcs://llama_weights_charlie/tokenizer.model \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/train_2.jsonl \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/val_2.jsonl \
#     --exp-name "llama_13B_test2" \
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
#     --model-mesh-shape 4 \
#     --fsdp \
#     --gradient-checkpoint \


# 4/19/2023

# tpu nlp-pod cl charlie-pod "source experiments/test_chess/train_tpu.sh"

# python -m llm_rl_scripts.chess.train_bc_llama \
#     PARAMS \
#     gcs://llama_weights_charlie/7B/ \
#     gcs://llama_weights_charlie/tokenizer.model \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/train.jsonl \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/val.jsonl \
#     --exp-name "llama_7B_test1" \
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
#     --model-mesh-shape 4 \
#     --fsdp \
#     --gradient-checkpoint \



# 4/18/2023

# tpu nlp-pod cl charlie-pod "source experiments/test_chess/train_tpu.sh"

# python -m llm_rl_scripts.chess.train_bc_llama \
#     PARAMS \
#     gcs://llama_weights_charlie/13B/ \
#     gcs://llama_weights_charlie/tokenizer.model \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/train.jsonl \
#     gcs://charlie-bucket2/datasets/chess_data/chess_data/val.jsonl \
#     --exp-name "llama_13B_test1" \
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
#     --model-mesh-shape 4 \
#     --fsdp \
#     --gradient-checkpoint \
