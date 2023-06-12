# from JaxSeq.utils import setup_experiment_save

# exp_name = "debug_setup_experiment_save"
# is_main_process = True
# outputs_path = "gcs://rail-tpus-isadora/llm-rl-outputs/chess/debugging/debug_setup_experiment_save"
# save_dir, exp_name = setup_experiment_save(
#         exp_name=exp_name, 
#         outputs_path=outputs_path, 
#         input_args={"arg": 1}, 
#         script__file__=__file__, 
#         is_main_process=is_main_process, 
#     )

from google.cloud import storage

def check_gcs_authentication():
    try:
        client = storage.Client()
        bucket = client.get_bucket('rail-tpus-isadora')
        print("Authentication successful!")
    except Exception as e:
        print("Authentication failed:", e)

check_gcs_authentication()

import os

def check_environment_variables():
    google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    gcloud_project = os.getenv('GCLOUD_PROJECT')

    print("GOOGLE_APPLICATION_CREDENTIALS:", google_application_credentials)
    print("GCLOUD_PROJECT:", gcloud_project)

check_environment_variables()