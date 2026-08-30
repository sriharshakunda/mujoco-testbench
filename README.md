# Agilex Piper 6-DOF Robotic Arm MuJoCo Testbench & LeRobot VLA Pipeline

A high-fidelity simulation environment, autonomous multi-modal demonstration collector, imitation learning benchmark, and HIL-SERL reinforcement learning pipeline for the **Agilex Piper 6-DOF manipulator** equipped with an analog parallel-jaw gripper.

---

## Table of Contents
1. [Setup Options (Docker vs. Virtual Environment)](#1-setup-options-docker-vs-virtual-environment)
2. [Data Collection](#2-data-collection)
   - [Manual Teleoperation Mode](#manual-teleoperation-mode)
   - [Automated Data Collection Mode](#automated-data-collection-mode)
   - [Tuning Autonomous Agent Parameters](#tuning-autonomous-agent-parameters)
3. [Policy Training](#3-policy-training)
   - [Fine-Tuning SmolVLA](#1-fine-tune-smolvla-vision-language-action-foundation-model)
   - [Training Diffusion Policy](#2-train-diffusion-policy-recommended-for-piper-arm)
   - [Training ACT Policy](#3-train-act-action-chunking-with-transformers)
   - [Resuming & Saving Intermediate Checkpoints](#4-resuming--saving-intermediate-checkpoints)
   - [Training on Cloud GPUs & Model Export](#5-training-on-cloud-gpus-runpod-lambda-aws-gcp--model-export)
4. [Policy Evaluation & Benchmarking](#4-policy-evaluation--benchmarking)
5. [Dataset Specifications (LeRobot v3.0)](#5-dataset-specifications-lerobot-v30)
6. [HIL-SERL Reinforcement Learning & DAgger Rollouts](#6-hil-serl-reinforcement-learning--dagger-rollouts)

---

## 1. Setup Options (Docker vs. Virtual Environment)

You can run every command in this pipeline either inside **Docker** or natively in a Python **Virtual Environment (`venv`)**.

### Option A: Virtual Environment (`venv`) Setup
```bash
# Create and configure virtual environment with site-packages compatibility patches
./setup_venv.sh

# Activate virtual environment
source venv/bin/activate
```

### Option B: Docker Setup
```bash
# Build Docker Image
./docker_run.sh --build

# Launch Interactive Sandbox
./docker_run.sh
```

---

## 2. Data Collection

### Manual Teleoperation Mode
Control the Piper arm manually using a **3Dconnexion SpaceMouse** or **Keyboard** to record demonstration trajectories:

- **Docker Command**:
  ```bash
  ./docker_run.sh --task "pick up the red block and place in blue bin" --data-dir data/my_manual_dataset
  ```
- **Venv Command**:
  ```bash
  python app.py --task "pick up the red block and place in blue bin" --data-dir data/my_manual_dataset
  ```

#### Keyboard Controls:
| Key | Action |
|---|---|
| `W / S` | Forward / Backward along table workspace ($+X / -X$) |
| `A / D` | Left / Right across table workspace ($+Y / -Y$) |
| `R / F` | Elevation Height ($+Z / -Z$) |
| `U / O` | Wrist Roll $\pm$ |
| `I / K` | Wrist Pitch $\pm$ |
| `J / L` | Wrist Yaw $\pm$ |
| `[ / ]` | Open / Close Gripper |
| `Space` / `C` | Start / Stop & Save Episode Recording |
| `N` | Discard Current Episode Buffer |
| `H` | Reset to Home Pose & Re-randomize Cubes |
| `Q` / `Esc` | Quit Teleoperation |

---

### Automated Data Collection Mode
Collect verified multi-modal demonstration episodes automatically. The autonomous controller generates realistic trajectories with tabletop cube pose randomization, picking the red cube and placing it into the target blue bin.

#### Interactive GUI Mode:
- **Docker Command**:
  ```bash
  ./docker_run.sh --auto-collect --num-episodes 10 --task "pick up the red block and place in blue bin" --data-dir data/my_auto_dataset
  ```
- **Venv Command**:
  ```bash
  python -m src.auto_collect --num-episodes 10 --task "pick up the red block and place in blue bin" --data-dir data/my_auto_dataset
  ```

#### Fast Headless Mode:
- **Docker Command**:
  ```bash
  ./docker_run.sh --auto-collect --num-episodes 50 --task "pick up the red block and place in blue bin" --data-dir data/my_auto_dataset --headless
  ```
- **Venv Command**:
  ```bash
  python -m src.auto_collect --num-episodes 50 --task "pick up the red block and place in blue bin" --data-dir data/my_auto_dataset --headless
  ```

---

### Tuning Autonomous Agent Parameters

The autonomous pick-and-place agent is driven by closed-loop differential Inverse Kinematics (`src/auto_collect.py`). You can tune trajectory waypoints, heights, speeds, and spawn randomization directly in `src/auto_collect.py` and `src/environment/env.py`:

| Parameter | Location | Default Value | Description |
|---|---|---|---|
| `TARGET_BIN_POS` | `src/auto_collect.py` | `[0.35, 0.32, 0.15]` | Target blue bin 3D center position |
| `HOVER_HEIGHT` | `src/auto_collect.py` | `0.28` (m) | Pre-grasp approach height hovering above cube |
| `GRASP_HEIGHT` | `src/auto_collect.py` | `0.165` (m) | Descent Z-level for closing finger pads on cube |
| `TRANSIT_HEIGHT` | `src/auto_collect.py` | `0.32` (m) | Lift height while carrying cube across table |
| `GRIPPER_OPEN / CLOSED` | `src/auto_collect.py` | `0.04` / `0.00` | Parallel gripper actuator opening/closing limits |
| `d_xy` Jitter Range | `src/environment/env.py` | `[-0.018, 0.018]` (m) | Tabletop cube spawn position noise range |

---

## 3. Policy Training

Train policies using official **Hugging Face LeRobot** (`lerobot-train` CLI integration or project helper scripts).

> 💡 **Automatic Directory Versioning**: Output directories auto-increment (`outputs/train/smolvla_piper`, `outputs/train/smolvla_piper_1`, `outputs/train/smolvla_piper_2`, etc.) so subsequent training runs never overwrite past checkpoints.

---

### 1. Fine-Tune SmolVLA (Vision-Language-Action Foundation Model)

- **Docker Command**:
  ```bash
  ./docker_run.sh --train \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type smolvla \
    --pretrained-path lerobot/smolvla_base \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Helper Command**:
  ```bash
  python -m src.train_policy \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type smolvla \
    --pretrained-path lerobot/smolvla_base \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Native CLI Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=local/dataset \
    --dataset.root=data/my_auto_dataset \
    --policy.type=smolvla \
    --steps=20000 \
    --batch_size=16 \
    --output_dir=outputs/train/smolvla_piper \
    --policy.push_to_hub=false
  ```

---

### 2. Train Diffusion Policy (Recommended for Piper Arm)

- **Docker Command**:
  ```bash
  ./docker_run.sh --train \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type diffusion \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Helper Command**:
  ```bash
  python -m src.train_policy \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type diffusion \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Native CLI Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=local/dataset \
    --dataset.root=data/my_auto_dataset \
    --policy.type=diffusion \
    --steps=20000 \
    --batch_size=16 \
    --output_dir=outputs/train/diffusion_piper \
    --policy.push_to_hub=false
  ```

---

### 3. Train ACT (Action Chunking with Transformers)

- **Docker Command**:
  ```bash
  ./docker_run.sh --train \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type act \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Helper Command**:
  ```bash
  python -m src.train_policy \
    --repo-id local/dataset \
    --dataset-root data/my_auto_dataset \
    --policy-type act \
    --steps 20000 \
    --batch-size 16
  ```
- **Venv Native CLI Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=local/dataset \
    --dataset.root=data/my_auto_dataset \
    --policy.type=act \
    --steps=20000 \
    --batch_size=16 \
    --output_dir=outputs/train/act_piper \
    --policy.push_to_hub=false
  ```

---

### 4. Resuming & Saving Intermediate Checkpoints

You can set `--save_freq` (or `--save-freq`) to save intermediate checkpoints and `--resume=true` to continue training from where you left off:

- **Save Intermediate Checkpoints Every 500 Steps**:
  ```bash
  lerobot-train \
    --dataset.repo_id=local/dataset \
    --dataset.root=data/my_auto_dataset \
    --policy.type=smolvla \
    --steps=2000 \
    --save_freq=500 \
    --batch_size=16 \
    --output_dir=outputs/train/smolvla_piper \
    --policy.push_to_hub=false
  ```

- **Resume & Extend Training (e.g. from 2,000 steps to 5,000 steps)**:
  ```bash
  lerobot-train \
    --dataset.repo_id=local/dataset \
    --dataset.root=data/my_auto_dataset \
    --policy.type=smolvla \
    --steps=5000 \
    --save_freq=500 \
    --batch_size=16 \
    --output_dir=outputs/train/smolvla_piper \
    --resume=true \
    --policy.push_to_hub=false
  ```

---

### 5. Training on Cloud GPUs (RunPod, Lambda, AWS, GCP) & Model Export

#### Option A: Hugging Face Hub Auto-Sync (Recommended)
Train in the cloud streaming your dataset directly from Hugging Face, and automatically push trained checkpoints back to Hugging Face Hub:

```bash
# On Cloud Instance:
export HF_TOKEN="hf_YourToken"
./docker_run.sh --train \
  --repo-id <HF_USER>/<DATASET_REPO_ID> \
  --policy-type diffusion \
  --steps 50000 \
  --push-to-hub \
  --policy-repo-id <HF_USER>/<POLICY_REPO_ID>
```

Download trained policy locally:
```bash
# On Local Machine:
huggingface-cli download <HF_USER>/<POLICY_REPO_ID> --local-dir outputs/train/diffusion_piper/checkpoints/last/pretrained_model
```

#### Option B: Direct File Sync via `rsync`
Sync trained policy checkpoint folder from cloud instance to local machine:
```bash
rsync -avz \
  ubuntu@<CLOUD_INSTANCE_IP>:/home/ubuntu/mujoco-testbench/outputs/train/ \
  ~/projects/mujoco-testbench/outputs/train/
```

---

## 4. Policy Evaluation & Benchmarking

Benchmark trained policy checkpoints in closed-loop MuJoCo simulation with full preprocessor normalization and action un-normalization.

### 1. Interactive / GUI Evaluation
- **Docker Command**:
  ```bash
  ./docker_run.sh --eval \
    --checkpoint outputs/train/smolvla_piper/checkpoints/000500/pretrained_model \
    --num-episodes 10
  ```
- **Venv Helper Command**:
  ```bash
  python -m src.evaluate_policy \
    --checkpoint outputs/train/smolvla_piper/checkpoints/000500/pretrained_model \
    --num-episodes 10
  ```
- **Venv Native CLI Command**:
  ```bash
  PYTHONPATH=. lerobot-eval \
    --policy.path=outputs/train/smolvla_piper/checkpoints/000500/pretrained_model \
    --env.type=piper \
    --eval.n_episodes=10
  ```

### 2. Fast Headless Evaluation with Video Saving
- **Docker Command**:
  ```bash
  ./docker_run.sh --eval \
    --checkpoint outputs/train/smolvla_piper/checkpoints/000500/pretrained_model \
    --num-episodes 10 \
    --max-steps 350 \
    --headless \
    --save-video \
    --video-dir eval_videos
  ```
- **Venv Helper Command**:
  ```bash
  python -m src.evaluate_policy \
    --checkpoint outputs/train/smolvla_piper/checkpoints/000500/pretrained_model \
    --num-episodes 10 \
    --max-steps 350 \
    --headless \
    --save-video \
    --video-dir eval_videos
  ```

---

## 5. Dataset Specifications (LeRobot v3.0)

Datasets are saved in the official **LeRobot v3.0** schema with chunked Parquet tables and H.264 compressed MP4 videos:

```
data/my_auto_dataset/
├── meta/
│   ├── info.json                      # Schema specification, codebase version v3.0
│   ├── stats.json                     # Normalization stats (min, max, mean, std)
│   ├── tasks.parquet                  # Language instructions
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet       # Per-episode chunk index metadata
├── data/
│   └── chunk-000/
│       └── file-000.parquet           # Timestamps, joint states, joint actions
└── videos/
    └── chunk-000/
        ├── observation.images.wrist/file-000.mp4       # Wrist RGB (H.264)
        └── observation.images.extrinsic/file-000.mp4   # Scene Overview (H.264)
```

---

## 6. HIL-SERL Reinforcement Learning & DAgger Rollouts

Human-In-The-Loop Sample-Efficient Reinforcement Learning (HIL-SERL) combines vision reward classifiers, online Soft Actor-Critic (SAC) RL, and real-time human interventions.

### 1. Train Vision Reward Classifier
Train a binary success detector on completed demonstration datasets:

- **Docker Command**:
  ```bash
  ./docker_run.sh --reward-classifier \
    --dataset-dir data/my_auto_dataset \
    --output-dir outputs/reward_classifier
  ```
- **Venv Command**:
  ```bash
  python -m src.reward_classifier \
    --dataset-dir data/my_auto_dataset \
    --output-dir outputs/reward_classifier
  ```

---

### 2. Launch HIL-SERL Online SAC Reinforcement Learning

- **Docker Command**:
  ```bash
  ./docker_run.sh --hil-serl --config configs/hilserl_piper.json
  ```
- **Venv Command**:
  ```bash
  python -m lerobot.rl.gym_manipulator --config configs/hilserl_piper.json
  ```

---

### 3. Collect DAgger Human Intervention Rollouts

- **Docker Command**:
  ```bash
  ./docker_run.sh --dagger --config configs/hilserl_piper.json
  ```
- **Venv Command**:
  ```bash
  lerobot-rollout --strategy.type=dagger --config configs/hilserl_piper.json
  ```
