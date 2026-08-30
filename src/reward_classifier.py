"""
Vision Reward Classifier for HIL-SERL Reinforcement Learning.
--------------------------------------------------------------
Trains a binary ResNet success detector (outputting 0.0 for incomplete, 1.0 for success)
from demonstration datasets (Parquet + MP4 videos) recorded in LeRobot v3.0 format.

Usage:
    python -m src.reward_classifier --dataset-dir data/my_auto_dataset --output-dir outputs/reward_classifier
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class GoalFrameDataset(Dataset):
    """Dataset wrapper extracting goal (success=1) and non-goal (initial=0) frames from LeRobot dataset."""

    def __init__(self, lerobot_dataset: LeRobotDataset, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        num_episodes = lerobot_dataset.num_episodes
        for ep_idx in range(num_episodes):
            ep_frames = lerobot_dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
            n_frames = len(ep_frames)
            if n_frames == 0:
                continue

            # Goal frames (last 10% of episode) -> Label 1.0
            goal_indices = range(int(n_frames * 0.9), n_frames)
            for idx in goal_indices:
                frame = ep_frames[idx]
                if "observation.images.extrinsic" in frame:
                    img = frame["observation.images.extrinsic"]
                    self.samples.append((img, 1.0))

            # Non-goal frames (first 20% of episode) -> Label 0.0
            non_goal_indices = range(0, min(int(n_frames * 0.2), n_frames))
            for idx in non_goal_indices:
                frame = ep_frames[idx]
                if "observation.images.extrinsic" in frame:
                    img = frame["observation.images.extrinsic"]
                    self.samples.append((img, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_data, label = self.samples[idx]
        if isinstance(img_data, torch.Tensor):
            img_tensor = img_data
        else:
            img_array = np.array(img_data, dtype=np.uint8)
            img_tensor = self.transform(img_array)
        return img_tensor, torch.tensor(label, dtype=torch.float32)


class ResNetRewardClassifier(nn.Module):
    """ResNet18 binary classifier for vision-based success probability estimation."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def train_reward_classifier(dataset_dir: str, output_dir: str, epochs: int = 5, lr: float = 1e-4):
    print(f"\n\033[1;34m[HIL-SERL Reward Classifier] Loading dataset from '{dataset_dir}'...\033[0m")
    lerobot_dataset = LeRobotDataset("local/dataset", root=Path(dataset_dir))

    dataset = GoalFrameDataset(lerobot_dataset)
    if len(dataset) == 0:
        print("\033[1;31mError: No valid frames found for reward classifier training.\033[0m")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetRewardClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\033[1;34m[HIL-SERL Reward Classifier] Training on {len(dataset)} samples for {epochs} epochs...\033[0m")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / "model.pt"
    torch.save(model.state_dict(), model_file)
    print(f"\033[1;32m✓ Reward classifier checkpoint saved to: {model_file}\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Train HIL-SERL Vision Reward Classifier")
    parser.add_argument("--dataset-dir", type=str, default="data/my_auto_dataset", help="Path to demonstration dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/reward_classifier", help="Path to save trained classifier checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_reward_classifier(args.dataset_dir, args.output_dir, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()

