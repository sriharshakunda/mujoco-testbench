"""
Official Hugging Face LeRobot Policy Training Pipeline for Agilex Piper Arm.
-----------------------------------------------------------------------------
Trains imitation learning policies (ACT, Diffusion Policy) using the official
Hugging Face `lerobot` Python library on recorded LeRobot datasets.

Features:
  - Official `LeRobotDataset` loading with action chunking delta timestamps
  - Multi-camera visual feature extraction (Wrist RGB, Depth, Scene, Topdown)
  - Official `make_policy` (ACT, Diffusion Policy)
  - Saves official Hugging Face pretrained models (`policy.save_pretrained`)

Usage:
  ./docker_run.sh --train --dataset-dir data/red_block_dataset --epochs 50
  ./docker_run.sh --train --policy-type diffusion --epochs 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def train_policy_lerobot(
    dataset_dir: str = "data/red_block_dataset",
    repo_id: str = "<HF_USER>/<DATASET_REPO_ID>",
    policy_type: str = "act",
    pretrained_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    chunk_size: int = 30,
    output_dir: str = "checkpoints/act_lerobot",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("      Hugging Face LeRobot Official Policy Training Pipeline")
    print("=" * 76)
    print(f"  Dataset Directory : \033[1;34m{dataset_dir}\033[0m")
    print(f"  Policy Type       : \033[1;35m{policy_type.upper()}\033[0m")
    if pretrained_path:
        print(f"  Pretrained Base   : \033[1;33m{pretrained_path}\033[0m")
    print(f"  Training Device   : \033[1;32m{device.upper()}\033[0m")
    print(f"  Epochs            : \033[1;36m{epochs}\033[0m")
    print(f"  Batch Size        : \033[1;36m{batch_size}\033[0m")
    print(f"  Action Horizon    : \033[1;36m{chunk_size} steps ({chunk_size / 30.0:.1f}s @ 30 FPS)\033[0m")
    print(f"  Output Directory  : \033[1;34m{output_dir}\033[0m")
    print("=" * 76 + "\n")

    # 1. Action chunking delta timestamps
    if policy_type.lower() == "diffusion":
        # Diffusion UNet requires temporal horizon to be a multiple of 8 (downsampling factor)
        if chunk_size % 8 != 0:
            chunk_size = ((chunk_size + 7) // 8) * 8
            print(f"  Note: Adjusted Diffusion horizon to multiple of 8: {chunk_size} steps")

    print(f"Loading LeRobotDataset from {dataset_dir}...")
    delta_timestamps = {
        "action": [i / 30.0 for i in range(chunk_size)],
    }
    if policy_type.lower() == "diffusion":
        delta_timestamps["observation.state"] = [0.0]

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_dir,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.04,
    )
    print(f"✓ Loaded {len(dataset)} total frames across {dataset.num_episodes} demonstration episodes.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Configure & Instantiate Policy using official LeRobot make_policy
    if policy_type.lower() == "act":
        if pretrained_path:
            print(f"Loading pretrained ACT policy from '{pretrained_path}'...")
            policy = ACTPolicy.from_pretrained(pretrained_path)
        else:
            cfg = ACTConfig(
                chunk_size=chunk_size,
                n_action_steps=chunk_size,
                device=device,
            )
            policy = make_policy(cfg, ds_meta=dataset.meta)
    elif policy_type.lower() == "diffusion":
        if pretrained_path:
            print(f"Loading pretrained Diffusion policy from '{pretrained_path}'...")
            policy = DiffusionPolicy.from_pretrained(pretrained_path)
        else:
            cfg = DiffusionConfig(
                horizon=chunk_size,
                n_action_steps=chunk_size,
                n_obs_steps=1,
                device=device,
            )
            policy = make_policy(cfg, ds_meta=dataset.meta)
    elif policy_type.lower() == "smolvla":
        # SmolVLA: Load pretrained Vision-Language backbone (SmolVLM2-500M) and adapt to Piper cameras
        print(f"Configuring SmolVLA with pretrained SmolVLM backbone (load_vlm_weights=True)...")
        cfg = SmolVLAConfig(
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            load_vlm_weights=True,
            train_expert_only=True,
            freeze_vision_encoder=True,
            device=device,
        )
        policy = make_policy(cfg, ds_meta=dataset.meta)
        if pretrained_path and pretrained_path != "lerobot/smolvla_base":
            print(f"Loading weights from custom checkpoint '{pretrained_path}'...")
            try:
                loaded = SmolVLAPolicy.from_pretrained(pretrained_path)
                policy.load_state_dict(loaded.state_dict(), strict=False)
            except Exception as e:
                print(f"  Note: {e}")
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}. Choose 'act', 'diffusion', or 'smolvla'.")
    policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"✓ Instantiated LeRobot {policy_type.upper()} Policy ({total_params / 1e6:.2f}M trainable parameters)\n")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")

    is_smolvla = policy_type.lower() == "smolvla"
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer if is_smolvla else None
    task_str = "place the red block in blue bin"

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if is_smolvla:
                # Map dataset camera keys to policy camera keys if needed
                policy_cams = list(policy.config.image_features.keys()) if hasattr(policy.config, "image_features") else []
                ds_cams = [k for k in batch.keys() if k.startswith("observation.images.") and "depth" not in k]
                if policy_cams and not any(k in batch for k in policy_cams):
                    for p_cam, d_cam in zip(policy_cams, ds_cams):
                        batch[p_cam] = batch[d_cam]

                cur_b = len(batch["observation.state"])
                tokens = tokenizer([task_str] * cur_b, return_tensors="pt", padding="max_length", max_length=48, truncation=True)
                batch["observation.language.tokens"] = tokens["input_ids"].to(device)
                batch["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)

            optimizer.zero_grad()
            loss, loss_dict = policy(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            # Save official Hugging Face pretrained model format
            policy.save_pretrained(str(output_path / "best_model"))

        star = " ★ Best" if is_best else ""
        print(f"  → Epoch {epoch:03d}/{epochs:03d}: Loss = {avg_loss:.4f}{star}")

    # Save final model checkpoint
    policy.save_pretrained(str(output_path / "final_model"))
    print("\n" + "=" * 76)
    print(f"  ✓ Training Complete! Hugging Face LeRobot Model Saved to: {output_path / 'best_model'}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Policy using Hugging Face LeRobot")
    parser.add_argument("--dataset-dir", type=str, default="data/red_block_dataset",
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--repo-id", type=str, default="<HF_USER>/<DATASET_REPO_ID>",
                        help="Hugging Face Dataset Repo ID")
    parser.add_argument("--policy-type", type=str, default="act", choices=["act", "diffusion", "smolvla"],
                        help="Policy architecture: 'act', 'diffusion', or 'smolvla' (default: act)")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path or Hugging Face repo ID for pretrained base model (e.g. lerobot/smolvla_base)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="Action chunk horizon (default: 30)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/act_lerobot",
                        help="Output directory for LeRobot model checkpoint")
    args = parser.parse_args()

    train_policy_lerobot(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        policy_type=args.policy_type,
        pretrained_path=args.pretrained_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
