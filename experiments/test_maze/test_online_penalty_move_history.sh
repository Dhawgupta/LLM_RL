export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
# sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256 --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 16 --num_pos_per_setup 4 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256_1roll_256pos --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 1 --num_pos_per_setup 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
export maze_name="double_t_maze"
#todo: fix this, this crashes when the trajectory length is too long!
for num_epochs in 1
do 
    for gamma in 0.9 0.95
    do 
        python llm_rl_scripts/maze/train_ppo_online.py HF gcs://rail-tpus-isadora/maze/maze_double_t_maze/double_t_maze_move_history_bc/model_141000 --n-rounds 100 --save-every-rounds 5 --exp-name ppo_"$maze_name"_move_history_penalty_num_epochs_"$num_epochs"kl_coeff_0.001_wd_0.01_lr_1e-6_gamma_"$gamma"_1024_rollouts --wandb-project ppo_"$maze_name" --model-mesh-shape 2 --data-mesh-shape 1 --fsdp-mesh-shape 1 --n-rollouts 1024 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-6 --maze-name "$maze_name" --describe-function "describe_observation" --init-kl-coef 0.001 --epochs "$num_epochs" --weight-decay 0.01 --reward-function "illegal_penalty_reward" --gamma "$gamma"
    done
done
# python llm_rl_scripts/maze/train_ppo_online.py HF gcs://rail-tpus-isadora/maze/maze_double_t_maze/double_t_maze_give_position_bc/model_141000 --n-rounds 100 --exp-name ppo_"$maze_name"_give_position_no_reranker_kl_0.5 --wandb-project ppo_"$maze_name" --model-mesh-shape 1 --data-mesh-shape 1 --fsdp-mesh-shape 2 --n-rollouts 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-6 --maze-name "$maze_name" --describe-function "describe_observation_give_position" --init-kl-coef 0.5
# python llm_rl_scripts/maze/train_ppo_online.py HF gcs://rail-tpus-isadora/maze/maze_umaze/umaze_give_position_bc/model_22000 --n-rounds 100 --exp-name ppo_"$maze_name"_give_position --wandb-project ppo_"$maze_name" --model-mesh-shape 2 --data-mesh-shape 1 --fsdp-mesh-shape 1 --n-rollouts 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-6 --maze-name "$maze_name" --describe-function "describe_observation_give_position"