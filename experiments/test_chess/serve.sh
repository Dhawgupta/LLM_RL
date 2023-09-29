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
conda activate LLM_RL

# 4/22/2023

python -m llm_rl_scripts.chess.eval_gpt2 \
    --host http://127.0.0.1:8026 \
    --data-file gcs://charlie-bucket2/datasets/chess_data/chess_data/val_2.jsonl \
    --max-iters 500 \
    --eos-token-id 13 \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.llama.llama_serve \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/chess_bc/llama_13B_test1.2023-04-21-22-57-24.225.e1424ae8e09711edb134b98208d53c85/last/ \
#     /shared/csnell/llama_weights/tokenizer.model \
#     --host 0.0.0.0 \
#     --port 8026 \

# python -m llm_rl_scripts.chess.eval_gpt2 \
#     --host http://127.0.0.1:8026 \
#     --data-file gcs://charlie-bucket2/datasets/chess_data/chess_data/val_2.jsonl \
#     --max-iters 500 \
#     --eos-token-id 198 \

# CUDA_VISIBLE_DEVICES=1 python -m examples_jaxseq.gpt2.gpt2_serve \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/chess_bc/gpt2_small_test2.2023-04-22-00-18-53.466.437948dce0a311edad1b000000acfe80/last/ \
#     --host 0.0.0.0 \
#     --port 8026 \

# 4/20/2023

# python -m llm_rl_scripts.chess.eval_gpt2 \
#     --host http://127.0.0.1:8026 \
#     --data-file gcs://charlie-bucket2/datasets/chess_data/chess_data/val.jsonl \

# CUDA_VISIBLE_DEVICES=7 python -m examples_jaxseq.gpt2.gpt2_serve \
#     PARAMS \
#     gcs://charlie-bucket2/JaxSeq2_outputs/chess_bc/gpt2_small_test1.2023-04-19-01-37-51.096.cc14f220de5211eda5e8d85ed384bd2b/last/ \
#     --host 0.0.0.0 \
#     --port 8026 \
