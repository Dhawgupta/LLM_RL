export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
# sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256 --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 16 --num_pos_per_setup 4 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256_1roll_256pos --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 1 --num_pos_per_setup 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
export maze_name="umaze"
#todo: fix this, this crashes when the trajectory length is too long!
python llm_rl_scripts/maze/train_ppo_online.py HF gcs://rail-tpus-isadora/maze/maze_umaze/umaze_move_history_bc/model_100000/ --n-rounds 100 --exp-name ppo_"$maze_name" --wandb-project ppo_"$maze_name" --model-mesh-shape 2 --n-rollouts 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5 --maze-name "$maze_name"