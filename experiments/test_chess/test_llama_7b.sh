source /home/isadoracw/miniconda3/bin/activate
conda activate LLM_RL
export PYTHONPATH=${PWD}:${PWD}/JaxSeq2/
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
sudo chmod -R 777 /home/isadoracw/LLM_RL
git config --global --add safe.directory /home/isadoracw/LLM_RL
export GOOGLE_APPLICATION_CREDENTIALS="/home/isadoracw/ILQL5/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
python -m pip install IPython
python -m pip install tyro
# python -m pip install -e .
python /home/isadoracw/LLM_RL/llm_rl_scripts/chess/train_bc_llama_3mill_games.py \
    PARAMS \
    gcs://rail-tpus-isadora-central2-b/llama-weights/ \
    gcs://rail-tpus-isadora-central2-b/llama-weights/tokenizer.model \
    gcs://rail-tpus-isadora-central2-b/eps_greedy_3mill/full_games.jsonl \
    --model-mesh-shape 1 --fsdp-mesh-shape 16 --data-mesh-shape -1 \
    --exp-name llama_3mill_games --wandb-project llama_7b_chess \
