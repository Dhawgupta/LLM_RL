conda activate LLM_RL

export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"

source secrets.sh

python -m llm_rl_scripts.maze.bc.partially_observed_bc \
    HF \
    gpt2 \
    gcs://rail-tpus-isadora/maze/data/double_t_maze_dialogue_history.jsonl \
    0.90 \
    --exp-name maze_partial_observation_test1 \
    --outputs-path gcs://rail-tpus-csnell-us/LLM_RL_outputs/maze/ \
    --data-mesh-shape 1 \
    --fsdp-mesh-shape -1 \
    --model-mesh-shape 1 \
    --use-wandb \
    --wandb-project bc_maze_LLM_RL \
    --epochs 20 \
    --train-bsize 128 \
    --max-input-length 384 \
    --max-output-length 4 \
    --log-every 256 \
    --eval-every-steps 256 \
    --eval-every-epochs 1 \
    --save-every-steps 256 \
    --save-every-epochs 1 \
    --save-at-end \
    --eval-loss-batches 64 \
    --generation-batches 64 \
    --eval-at-beginning


