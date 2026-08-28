"""
Official Hugging Face LeRobot Policy Training Launcher for Agilex Piper Arm.
-----------------------------------------------------------------------------
Wraps the official Hugging Face `lerobot-train` CLI to ensure training utilizes
official pre/postprocessors, EMA, learning rate warmups, step-based schedules,
and automated checkpoint serialization.

Usage:
  python -m src.train_policy --repo-id user/dataset_name --policy-type act --steps 50000
  python -m src.train_policy --repo-id user/dataset_name --policy-type diffusion --steps 20000
  python -m src.train_policy --repo-id user/dataset_name --policy-type smolvla --pretrained-path lerobot/smolvla_base --steps 20000
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional


def get_total_frames(dataset_root: str) -> Optional[int]:
    """Read total_frames from dataset info.json if available."""
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
                return info.get("total_frames")
        except Exception:
            pass
    return None


def get_unique_output_dir(base_dir: str) -> str:
    """Auto-increment output directory (e.g. dir_1, dir_2) if directory exists and resume is False."""
    if not os.path.exists(base_dir):
        return base_dir
    counter = 1
    while os.path.exists(f"{base_dir}_{counter}"):
        counter += 1
    return f"{base_dir}_{counter}"


def launch_lerobot_train(
    repo_id: str,
    dataset_root: str = "data/red_block_dataset",
    policy_type: str = "act",
    pretrained_path: Optional[str] = None,
    steps: int = 20000,
    batch_size: int = 16,
    tolerance_s: float = 0.04,
    output_dir: Optional[str] = None,
    push_to_hub: bool = False,
    policy_repo_id: Optional[str] = None,
    device: str = "cuda",
    save_freq: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
):
    if output_dir is None:
        output_dir = f"outputs/train/{policy_type}_piper"

    if not resume:
        output_dir = get_unique_output_dir(output_dir)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}",
    ]

    # Only append --dataset.root if directory exists locally
    if dataset_root and dataset_root.lower() not in ["none", "null", ""] and os.path.exists(dataset_root):
        cmd.append(f"--dataset.root={dataset_root}")

    cmd.extend([
        "--dataset.use_imagenet_stats=false",
        f"--tolerance_s={tolerance_s}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--output_dir={output_dir}",
        f"--job_name={policy_type}_piper_training",
        f"--policy.device={device}",
    ])

    if save_freq is not None:
        cmd.append(f"--save_freq={save_freq}")

    if resume:
        cmd.append("--resume=true")

    cmd.append(f"--policy.type={policy_type.lower()}")

    if policy_type.lower() == "smolvla":
        base_path = pretrained_path or "lerobot/smolvla_base"
        cmd.append(f"--policy.pretrained_path={base_path}")
        rename_map = '{"observation.images.wrist": "observation.images.camera1", "observation.images.extrinsic": "observation.images.camera2", "observation.images.topdown": "observation.images.camera3"}'
        cmd.append(f"--rename_map={rename_map}")
    elif pretrained_path:
        cmd.append(f"--policy.pretrained_path={pretrained_path}")

    if push_to_hub:
        if not policy_repo_id:
            raise ValueError("--policy-repo-id must be provided when --push-to-hub is enabled.")
        cmd.append("--policy.push_to_hub=true")
        cmd.append(f"--policy.repo_id={policy_repo_id}")
    else:
        cmd.append("--policy.push_to_hub=false")

    print("\n" + "=" * 76)
    print("      Hugging Face LeRobot Official CLI Training Launcher")
    print("=" * 76)
    print(f"  Dataset Repo ID : {repo_id}")
    print(f"  Policy Type     : {policy_type.upper()}")
    print(f"  Training Steps  : {steps}")
    print(f"  Batch Size      : {batch_size}")
    print(f"  Tolerance (s)   : {tolerance_s}")
    print(f"  Output Dir      : {output_dir}")
    print(f"  Full Command    : {' '.join(cmd)}")
    print("=" * 76 + "\n")

    if dry_run:
        print("[Dry Run] Command generated successfully.")
        return cmd

    # Suppress verbose warnings in subprocess
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    # Run subprocess while suppressing verbose config dictionary dumps
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    suppress = False
    for line in process.stdout:
        # Detect start of giant config dictionary dump
        if "ot_train.py:" in line and " {'batch_size':" in line:
            suppress = True
            continue
        # Resume normal output when config dump finishes
        if suppress:
            if "Creating dataset" in line or "Creating policy" in line or "End of training" in line or "Traceback" in line:
                suppress = False
                sys.stdout.write(line)
                sys.stdout.flush()
        else:
            sys.stdout.write(line)
            sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Train Policy using Official Hugging Face LeRobot CLI")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face Dataset Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--dataset-root", type=str, default="data/red_block_dataset",
                        help="Local path to dataset (default: data/red_block_dataset)")
    parser.add_argument("--policy-type", type=str, default="act", choices=["act", "diffusion", "smolvla"],
                        help="Policy architecture: 'act', 'diffusion', or 'smolvla' (default: act)")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path or Hugging Face repo ID for pretrained base model")
    parser.add_argument("--epochs", type=float, default=None,
                        help="Number of training epochs (automatically converted to steps based on dataset size)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of training steps (default: 20000 if --epochs is not specified)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Mini-batch size (default: 16)")
    parser.add_argument("--tolerance-s", type=float, default=0.04,
                        help="Dataset timestamp tolerance in seconds (default: 0.04)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for LeRobot model checkpoint")
    parser.add_argument("--push-to-hub", action="store_true", default=False,
                        help="Push trained policy to Hugging Face Hub")
    parser.add_argument("--policy-repo-id", type=str, default=None,
                        help="Hugging Face Repo ID for trained policy output (required if --push-to-hub is set)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run training on (default: cuda)")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Interval in steps at which intermediate checkpoints are saved (default: 20000)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from an existing output directory checkpoint")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print command without executing")
    args = parser.parse_args()

    steps = args.steps
    if args.epochs is not None:
        total_frames = get_total_frames(args.dataset_root)
        if total_frames is not None:
            steps = max(1, int((total_frames / args.batch_size) * args.epochs))
            print(f"[Epoch Converter] Configured {args.epochs} epoch(s) @ batch_size={args.batch_size} ({total_frames} total frames) -> {steps} steps.")
        else:
            steps = max(1, int(1060 * args.epochs))
            print(f"[Epoch Converter] Configured {args.epochs} epoch(s) -> {steps} steps.")
    elif steps is None:
        steps = 20000

    launch_lerobot_train(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        policy_type=args.policy_type,
        pretrained_path=args.pretrained_path,
        steps=steps,
        batch_size=args.batch_size,
        tolerance_s=args.tolerance_s,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        policy_repo_id=args.policy_repo_id,
        device=args.device,
        save_freq=args.save_freq,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
