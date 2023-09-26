conda activate LLM_RL

export GCLOUD_PROJECT="rail-tpus"
export GCLOUD_TOKEN_PATH="${HOME}/.config/gcloud/rail-tpus.json"

source secrets.sh

# python -m llm_rl_scripts.maze.eval.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-isadora/maze/maze_double_t_maze/double_t_maze_give_position_bc_submazes/model_145001/LLM_RL/ \
#     --policy-n-rollouts 8 \
#     --policy-bsize 8 \
#     --no-policy-do-sample \

# python -m llm_rl_scripts.maze.eval.eval_bc \
#     PARAMS \
#     gcs://rail-tpus-isadora/maze/maze_double_t_maze/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward.2023-09-18-03-50-20.916.7d5547b055d611ee9749e351425a3ca0/round_24/policy \
#     --policy-n-rollouts 8 \
#     --policy-bsize 8 \
#     --no-policy-do-sample \

# python -m llm_rl_scripts.maze.eval.eval_ilql \
#     PARAMS \
#     gcs://rail-tpus-isadora/maze/maze_double_t_maze/llm_rl_ilql_submazes_double_t_maze/llm_rl_ilql_submazes_double_t_maze.2023-09-19-22-39-17.058.5da20b0c573d11eeb372452a3d0fbbc7/last \
#     PARAMS \
#     gcs://rail-tpus-isadora/maze/maze_double_t_maze/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward.2023-09-18-03-50-20.916.7d5547b055d611ee9749e351425a3ca0/round_24/policy \
#     --policy-n-rollouts 1 \
#     --policy-bsize 1 \
#     --policy-beta 1.0 \
#     --use-reranker-for-reward-eval \
#     # --no-policy-do-sample \

python -m llm_rl_scripts.maze.eval.eval_mc \
    PARAMS \
    gcs://rail-tpus-isadora/maze/maze_double_t_maze/llm_rl_mc_returns_submazes_double_t_maze/llm_rl_mc_returns_submazes_double_t_maze.2023-09-20-00-36-58.687.ceb122be574d11eebb0903f6f25258ac/last \
    PARAMS \
    gcs://rail-tpus-isadora/maze/maze_double_t_maze/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward/submazes_diff_start_ppo_double_t_maze_give_position_penalty_num_epochs_1kl_coeff_0.001_wd_0.01_lr_1e-5_gamma_0.95_standard_reward.2023-09-18-03-50-20.916.7d5547b055d611ee9749e351425a3ca0/round_24/policy \
    --policy-n-rollouts 1 \
    --policy-bsize 1 \
    --policy-beta 1.0 \
    --use-reranker-for-reward-eval \
    # --no-policy-do-sample \

