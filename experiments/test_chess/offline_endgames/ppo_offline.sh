export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"

sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_ppo_gpt2_offline_endgames.py HF outputs/chess/test_bc_shuffled2/model/ --exp-name endgames_offline_lr_1e-4 --wandb-project ppo_endgames --ppo-data-bsize 8 --train-bsize 8 --grad-accum-steps 16 --model-mesh-shape 4 --lr 1e-4 --num_pos_per_setup 4 --n-rollouts 8
