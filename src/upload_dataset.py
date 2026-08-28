"""
Hugging Face Dataset Uploader for LeRobot VLA Pipeline.
------------------------------------------------------
Uploads recorded LeRobot v2.0 datasets directly to Hugging Face Hub.

Usage:
  python -m src.upload_dataset --data-dir data/my_vla_dataset --repo-id <hf_user>/<dataset_name>
"""

import os
import sys
import argparse
from pathlib import Path


def upload_to_huggingface(data_dir: str, repo_id: str, private: bool = False, token: str = None):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("\033[1;31mError: huggingface_hub is not installed. Run: pip install huggingface_hub\033[0m")
        sys.exit(1)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\033[1;31mError: Directory '{data_dir}' does not exist.\033[0m")
        sys.exit(1)

    hf_token = token or os.environ.get("HF_TOKEN") or None

    api = HfApi(token=hf_token)

    print(f"\n\033[1;34m[Hugging Face Upload] Preparing repository '{repo_id}' (private={private})...\033[0m")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=hf_token)
        print(f"\033[1;32m✓ Repository '{repo_id}' verified/created.\033[0m")
    except Exception as e:
        print(f"\033[1;31mFailed to create/access repo: {e}\033[0m")
        print("Please make sure you are logged in via 'huggingface-cli login' or have HF_TOKEN set.")
        sys.exit(1)

    print(f"\033[1;34m[Hugging Face Upload] Uploading '{data_dir}' to 'https://huggingface.co/datasets/{repo_id}'...\033[0m")

    try:
        api.upload_folder(
            folder_path=str(data_path),
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns=["*"],
            token=hf_token,
        )
        print(f"\n\033[1;32m========================================================================")
        print(f"  ✓ Successfully uploaded dataset to:")
        print(f"    https://huggingface.co/datasets/{repo_id}")
        print(f"========================================================================\033[0m\n")
    except Exception as e:
        print(f"\033[1;31mUpload failed: {e}\033[0m")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot VLA Dataset to Hugging Face Hub")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to recorded dataset directory (e.g. data/my_vla_dataset)")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repo ID (e.g. username/piper_cube_pick_place)")
    parser.add_argument("--private", action="store_true",
                        help="Make the Hugging Face dataset private")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token (optional if already logged in)")
    args = parser.parse_args()

    upload_to_huggingface(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()

