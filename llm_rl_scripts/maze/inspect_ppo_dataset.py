from google.cloud import storage
import pickle as pkl
import io 

client = storage.Client()


bucket_name = "rail-tpus-isadora"
blob_name = "maze/maze_umaze/ppo_umaze_give_position/ppo_umaze_give_position.2023-07-31-21-39-29.924.bae160da2fea11eeacd117c98577a39c/data_saves/6/text_trajectory_chains.pkl"

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)

pickle_data = io.BytesIO()
blob.download_to_file(pickle_data)
pickle_data.seek(0)  # Reset the stream position to the beginning

# Load the pickle data into a Python object
loaded_data = pkl.load(pickle_data)
print(loaded_data)