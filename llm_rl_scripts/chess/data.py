import numpy as np
from google.cloud import storage
import os
import io
from LLM_RL.environment import Text, TextTrajectory, TextTrajectoryChain
import json
from llm_rl_scripts.chess.env import preprocess_move, preprocess_state

# cwd = os.getcwd()
# key_path = os.path.join(cwd, "rail-tpus.json")

# Replace "path/to/service-account-key.json" with the actual path to your service account key file
client = storage.Client.from_service_account_json("/nfs/nfs1/users/isadoracw/rail-tpus.json")

bucket_name = "rail-tpus-isadora"
blob_name = "queen_rook_unopposed/queen_rook_unopposed/rl_train.jsonl"

def get_data_from_bucket(bucket_name, blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)

    blob_data = blob.download_as_text()
    blob_data = blob_data.split("\n")
    return blob_data

def chess_text_transition_from_json(data, scaling=1):
    # lst = list(f)
    for obj in data:
        # print(obj)
        if obj == "":
            continue
        result =  json.loads(obj)
        from_state = Text(preprocess_state(result["from_state"]), False)
        action = Text(preprocess_move(result["action"]), True) 
        to_state = Text(preprocess_state(result["to_state"]), False)
        next_action = Text(preprocess_move(result["next_action"]), True)

        curr_trajectory = TextTrajectory([from_state, action], [0, scaling*result["reward"]], result["done"])
        next_trajectory = TextTrajectory([to_state, next_action], [0, scaling*result["next_reward"]], result["next_done"])
        yield TextTrajectoryChain(curr_trajectory, [next_trajectory])

data = get_data_from_bucket(bucket_name, blob_name)
data = list(chess_text_transition_from_json(data))
print(data[:10])