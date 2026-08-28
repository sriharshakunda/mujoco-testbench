import argparse
from huggingface_hub import HfApi


def push_dataset():
    parser = argparse.ArgumentParser(description="Upload dataset folder to Hugging Face Hub")
    parser.add_argument("--folder-path", type=str, required=True, help="Path to local dataset directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Dataset Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--commit-message", type=str, default="Upload LeRobot dataset", help="Commit message")
    args = parser.parse_args()

    api = HfApi()
    api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"Pushed '{args.folder_path}' to Hugging Face Hub repo '{args.repo_id}' successfully!")


if __name__ == "__main__":
    push_dataset()
