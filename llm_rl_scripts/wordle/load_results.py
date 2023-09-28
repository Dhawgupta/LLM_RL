import os
import json
import math
from JaxSeq.bucket_manager import open_with_bucket as open

# BC: charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_test3.2023-09-22-21-53-58.938.88bf2e58599211ee812d4554a3c5cde2/last
# 50% BC: charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_50.2023-09-22-22-01-52.694.a32076b6599311eeaa2d5bbde740719c/last
# 30% BC: charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_30.2023-09-23-05-02-18.636.5ef5bfd859ce11eeaa2d5bbde740719c/last
# 10% BC: charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_10.2023-09-23-09-14-33.106.9bcea4e259f111eeaa2d5bbde740719c/last
# PPO: waiting
# ILQL: charlie-bucket2/LLM_RL_outputs/wordle/worlde_gpt2_ilql_test1/wordle_gpt2_ilql_test1.2023-09-24-23-55-18.774.d0b16fba5b3511eeaa2d5bbde740719c/epoch_8
# MC: charlie-bucket2/LLM_RL_outputs/wordle/worlde_gpt2_mc_test1/wordle_gpt2_mc_test1.2023-09-24-23-22-57.716.4bbb75345b3111ee812d4554a3c5cde2/epoch_3

if __name__ == "__main__":
    BC_PATHS = {
        'BC': 'gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_test3.2023-09-22-21-53-58.938.88bf2e58599211ee812d4554a3c5cde2/last',
        '50%_BC': 'gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_50.2023-09-22-22-01-52.694.a32076b6599311eeaa2d5bbde740719c/last',
        '30%_BC': 'gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_30.2023-09-23-05-02-18.636.5ef5bfd859ce11eeaa2d5bbde740719c/last',
        '10%_BC': 'gcs://charlie-bucket2/JaxSeq2_outputs/wordle_bc/wordle_gpt2_config_test1_filtered_10.2023-09-23-09-14-33.106.9bcea4e259f111eeaa2d5bbde740719c/last',
    }

    ILQL_PATHS = {
        'ilql': 'gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_gpt2_ilql_test1/wordle_gpt2_ilql_test1.2023-09-24-23-55-18.774.d0b16fba5b3511eeaa2d5bbde740719c/epoch_8',
    }

    MC_PATHS = {
        'mc': 'gcs://charlie-bucket2/LLM_RL_outputs/wordle/worlde_gpt2_mc_test1/wordle_gpt2_mc_test1.2023-09-24-23-22-57.716.4bbb75345b3111ee812d4554a3c5cde2/epoch_3',
    }

    for name, path in BC_PATHS.items():
        with open(os.path.join(path, 'eval_bc_greedy', 'interactions_summary.json'), 'r') as f:
            item_greedy = json.load(f)
        with open(os.path.join(path, 'eval_bc_sample', 'interactions_summary.json'), 'r') as f:
            item_sample = json.load(f)
        print(name, 'greedy:', f"{item_greedy['reward']['mean']} +- {item_greedy['reward']['std'] / math.sqrt(4096)}")
        print(name, 'sample:', f"{item_sample['reward']['mean']} +- {item_sample['reward']['std'] / math.sqrt(4096)}")
    
    for name, path in ILQL_PATHS.items():
        for beta in [1, 2, 4, 8, 16, 32, 64, 128]:
            with open(os.path.join(path, f'eval_ilql_beta{beta}_greedy', 'interactions_summary.json'), 'r') as f:
                item_greedy = json.load(f)
            with open(os.path.join(path, f'eval_ilql_beta{beta}_sample', 'interactions_summary.json'), 'r') as f:
                item_sample = json.load(f)
            print(name, f'greedy_{beta}:', f"{item_greedy['reward']['mean']} +- {item_greedy['reward']['std'] / math.sqrt(4096)}")
            print(name, f'sample_{beta}:', f"{item_sample['reward']['mean']} +- {item_sample['reward']['std'] / math.sqrt(4096)}")

    for name, path in MC_PATHS.items():
        for beta in [1, 2, 4, 8, 16, 32, 64, 128]:
            with open(os.path.join(path, f'eval_mc_beta{beta}_greedy', 'interactions_summary.json'), 'r') as f:
                item_greedy = json.load(f)
            with open(os.path.join(path, f'eval_mc_beta{beta}_sample', 'interactions_summary.json'), 'r') as f:
                item_sample = json.load(f)
            print(name, f'greedy_{beta}:', f"{item_greedy['reward']['mean']} +- {item_greedy['reward']['std'] / math.sqrt(4096)}")
            print(name, f'sample_{beta}:', f"{item_sample['reward']['mean']} +- {item_sample['reward']['std'] / math.sqrt(4096)}")
