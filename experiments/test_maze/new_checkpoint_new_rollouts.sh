export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
# sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256 --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 16 --num_pos_per_setup 4 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_online_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name ppo_online_endgames_lr1e-5_bsize256_1roll_256pos --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 1 --num_pos_per_setup 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5
export maze_name="double_t_maze"
python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/maze/train_ppo_online.py HF gcs://rail-tpus-isadora/maze/maze_double_t_maze/double_t_maze_give_position_bc_submazes/model_145001 --n-rounds 100 --save-every-rounds 5 --exp-name submazes_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95 --wandb-project ppo_double_t_maze --model-mesh-shape 1 --data-mesh-shape 2 --fsdp-mesh-shape 1 --n-rollouts 64 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 16 --lr 1e-5 --maze-name double_t_maze --describe-function describe_observation_give_position --init-kl-coef 0.001 --epochs 1 --weight-decay 0.01 --reward-function illegal_penalty_reward --gamma 0.95