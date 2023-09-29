export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/chess/train_mc_returns.py HF outputs/chess/test_bc_shuffled2/model/ --exp-name mc_returns_from_ppo_data --wandb-project mc_ppo_data --model-mesh-shape 4 --use-wandb --train-bsize 8 --grad-accum-steps 16