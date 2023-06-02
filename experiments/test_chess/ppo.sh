export TOKENIZERS_PARALLELISM=false
# 2 cores per process
alias TPU0="TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
alias TPU1="TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477"
alias TPU2="TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"
alias TPU3="TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479"
# 4 cores per process
alias TPU01="TPU_VISIBLE_DEVICES=0,1 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
alias TPU23="TPU_VISIBLE_DEVICES=2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"

conda init bash
conda activate LLM_RL
TPU1 python ~/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_chkpts --wandb-project ppo_endgames --model-mesh-shape 2 --n-rollouts 16 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 4