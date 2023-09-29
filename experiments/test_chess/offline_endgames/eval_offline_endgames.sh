export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_offline_lr1e-7 --checkpoint-dir gcs://rail-tpus-isadora/llm-rl-outputs/chess/endgames_offline_lr_1e-7/endgames_offline_lr_1e-7.2023-06-12-20-15-04.787.d1950466095d11ee8b308de166d61c57/best

# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_random_pos_16_round_99 --random-positions
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_online_lr1e-6_test_positions --checkpoint-dir /home/isadoracw/isadoracw/LLM_RL/outputs/chess/lr1e-6_ppo_online_endgames_queen_rook_save/lr1e-6_ppo_online_endgames_queen_rook_save.2023-06-06-19-22-40.564.8100307e049f11ee8b308de166d61c57/last
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_online_lr1e-6_test_positions --checkpoint-dir /home/isadoracw/isadoracw/LLM_RL/outputs/chess/endgames_offline_lr_1e-7_wd_0.01/endgames_offline_lr_1e-7_wd_0.01.2023-06-05-03-59-55.879.6ea9e230035511eeae8329568c12d0ad/step_77824/
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_random_pos_16_round_79 --random-positions --checkpoint-dir /home/isadoracw/isadoracw/LLM_RL/outputs/chess/lr1e-6_ppo_online_endgames_queen_rook/lr1e-6_ppo_online_endgames_queen_rook.2023-06-05-07-13-26.269.76fee0fa037011eeb99cbd216c6583a9/round_79
# python /nfs/nfs1/users/isadoracw/LLM_RL/llm_rl_scripts/eval_chess/eval_ppo.py --model-mesh-shape 2 --exp-name ppo_random_pos_16_round_69 --random-positions --checkpoint-dir /home/isadoracw/isadoracw/LLM_RL/outputs/chess/lr1e-6_ppo_online_endgames_queen_rook/lr1e-6_ppo_online_endgames_queen_rook.2023-06-05-07-13-26.269.76fee0fa037011eeb99cbd216c6583a9/round_69