from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="data/redcube_picknplace_v3",
    repo_id="<HF_USER>/<DATASET_REPO_ID>",
    repo_type="dataset",
    commit_message="Migrate dataset to LeRobot v3.0 schema"
)
print("Pushed to Hugging Face Hub successfully!")
