export TOKENIZERS_PARALLELISM=false
sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*

python /home/isadoracw/isadoracw/LLM_RL/~/isadoracw/LLM_RL/outputs/chess/ppo_online_endgames_queen_rook/ppo_online_endgames_queen_rook.2023-06-04-22-46-25.986.a317fd12032911ee9fce87d7217c0314/config.py HF outputs/chess/test_bc_shuffled2/model/ --n-rounds 100 --exp-name reproduce_ppo_online_endgames_queen_rook --wandb-project ppo_endgames --model-mesh-shape 4 --n-rollouts 16 --ppo-data-bsize 8 --use-wandb --train-bsize 8 --grad-accum-steps 4