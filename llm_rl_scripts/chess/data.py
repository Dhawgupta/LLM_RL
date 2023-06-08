import numpy as np
from google.cloud import storage
import os
import io
from LLM_RL.environment import Text, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain
import json
from llm_rl_scripts.chess.env import large_piece_random_endgame, preprocess_move, preprocess_state, preprocess_state_og
import random
import chess
from IPython import embed
from tqdm.auto import tqdm
import pickle
import glob

# cwd = os.getcwd()
# key_path = os.path.join(cwd, "rail-tpus.json")

# Replace "path/to/service-account-key.json" with the actual path to your service account key file
client = storage.Client.from_service_account_json("/nfs/nfs1/users/isadoracw/rail-tpus.json")

bucket_name = "rail-tpus-isadora"
blob_name = "queen_rook_unopposed/queen_rook_unopposed/train_unshuffled.jsonl"

def get_data_from_bucket(bucket_name, blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(blob_name)

    blob_data = blob.download_as_text()
    blob_data = blob_data.split("\n")
    return blob_data

def get_random_positions_not_in_test(bucket_name=bucket_name, blob_name=blob_name, num_pos_per_setup=4):
    test_positions = get_data_from_bucket(bucket_name, blob_name)
    test_positions = test_positions[:500]
    test_positions = [position.replace("\n", "").replace("\"", "") for position in test_positions]
    
    total_positions = []
    for setup in ["kQK", "kRK", "kQRK", "kRRK"]:
        random_positions = []
        while len(random_positions) < num_pos_per_setup:
            random_position = large_piece_random_endgame(setup)
            if random_position not in test_positions:
                random_positions.append(random_position)
        total_positions.extend(random_positions)
    
    return total_positions

def get_saved_text_chains(data_saves_dir):
    
    # find all directories in data_saves_dir
    directories = glob.glob(data_saves_dir + "/*")
    # get all text_trajectory_chains.pkl files and concatenate them
    total_text_trajectory_chains = []
    for directory in directories:
        path = str(directory) + "/text_trajectory_chains.pkl"
        # load pickle file and print results
        with open(path, 'rb') as f:
            pickle_data = pickle.load(f)
            total_text_trajectory_chains += pickle_data
    return total_text_trajectory_chains

# blob_name = "queen_rook_unopposed/queen_rook_unopposed/test_positions.jsonl"
# print(get_random_positions_not_in_test(blob_name=blob_name))
    
# def chess_text_trajectory_chain_from_json(data, scaling=1):
#     # lst = list(f)
#     for obj in data:
#         # print(obj)
#         if obj == "":
#             continue
#         result =  json.loads(obj)
#         from_state = Text(preprocess_state(result["from_state"]), False)
#         action = Text(preprocess_move(result["action"]), True) 
#         to_state = Text(preprocess_state(result["to_state"]), False)
#         next_action = Text(preprocess_move(result["next_action"]), True)

#         curr_trajectory = TextTrajectory([from_state, action], [0, scaling*result["reward"]], result["done"])
#         next_trajectory = TextTrajectoryChain(TextTrajectory([to_state, next_action], [0, scaling*result["next_reward"]], result["next_done"]))
#         yield TextTrajectoryChain(curr_trajectory, next_trajectory)

def chess_text_trajectory_chain_from_json(data, scaling=1):
    idx = 0
    text_trajectory_chains = []
    while idx < len(data):
        trajectories = []
        done = False
        while not done and idx < len(data):
            if data[idx] == "":
                # print("here!")
                # embed()
                idx += 1
                break
            result = json.loads(data[idx])
            state = Text(preprocess_state_og(result["from_state"]), False)
            action = Text(preprocess_move(result["action"]), True)
            trajectory = TextTrajectory([state, action], [0, scaling*result["reward"]], result["done"])
            trajectories.append(trajectory)
            done = result["done"]
            idx += 1
            
            if len(trajectories) == 200:
                break
        
        if not trajectories:
            break
        chain = None
        for text_trajectory in trajectories[::-1]:
            chain = TextTrajectoryChain(
                text_trajectory=text_trajectory, 
                next=chain, 
            )
        # print(chain)
        text_trajectory_chains.append(chain)
    random.shuffle(text_trajectory_chains)
    return text_trajectory_chains
            # if not result["done"]:
            # data.append(result) 

def chess_trajectory_chain_from_npy(actions, states, done, reward):
    text_trajectory_chains = []
    init_state = chess.Board().fen()
    print(len(actions))
    for game_idx in tqdm(range(len(actions))):
        trajectories = []
        move_idx = 0
        d = False
        while move_idx + 1 < 200 and not d:
            if move_idx == 0:
                state = Text(preprocess_state_og(init_state), False)
            else:
                state = Text(preprocess_state_og(states[game_idx][move_idx - 1]), False)
            action = Text(preprocess_move(actions[game_idx][move_idx]), True)
            if move_idx == 199:
                d = True
            else:
                d = done[game_idx][move_idx]
            trajectory = TextTrajectory([state, action], [0, reward[game_idx][move_idx]], d)
            trajectories.append(trajectory)
            move_idx += 1
        chain = None
        for text_trajectory in trajectories[::-1]:
            chain = TextTrajectoryChain(
                text_trajectory=text_trajectory, 
                next=chain, 
            )
        text_trajectory_chains.append(chain)
        
    random.shuffle(text_trajectory_chains)
    return text_trajectory_chains

def get_dataset(dataset_path):
    actions = np.load(os.path.join(dataset_path, "actions.npy"), mmap_mode="r")
    states = np.load(os.path.join(dataset_path, "states.npy"), mmap_mode="r")
    done = np.load(os.path.join(dataset_path, "done.npy"), mmap_mode="r")
    reward = np.load(os.path.join(dataset_path, "reward.npy"), mmap_mode="r")
    return actions, states, done, reward

dataset_path = os.path.join("/nfs/nfs1/users/isadoracw/ILQL5/src/environments/chess/complete_background_generated/")
actions, states, done, reward = get_dataset(dataset_path)
text_trajectory_chains = chess_trajectory_chain_from_npy(actions[:100], states, done, reward)
# print(text_trajectory_chains[:10])
print(len(text_trajectory_chains))
# print(text_trajectory_chains[:10])
# # data = get_data_from_bucket(bucket_name, blob_name)
# chains = chess_text_trajectory_chain_from_json(data)
# # chains[:10]
# print(chains[:10])
# # token_trajectory_chains = [
#             TokenTrajectoryChain.from_text_trajectory_chain(
#                 item, 
#                 self.tokenizer, 
#                 token_process=token_process, 
#             ) for item in data
#         ]
# print(data[:10])