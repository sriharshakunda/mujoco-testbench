# Agilex Piper 6-DOF Robotic Arm MuJoCo Testbench & LeRobot VLA Pipeline

A high-fidelity simulation environment, autonomous multi-modal demonstration collector, and imitation learning benchmark for the **Agilex Piper 6-DOF manipulator** equipped with an analog parallel-jaw gripper.

---

## Table of Contents
1. [Docker Setup & Compilation](#1-docker-setup--compilation)
2. [Data Collection](#2-data-collection)
   - [Manual Teleoperation Mode](#manual-teleoperation-mode)
   - [Automated Data Collection Mode](#automated-data-collection-mode)
   - [Tuning Autonomous Agent Parameters](#tuning-autonomous-agent-parameters)
3. [Policy Training](#3-policy-training)
   - [Training on Local Machine](#training-on-local-machine)
   - [Training in Notebooks / Cloud GPUs](#training-in-notebooks--cloud-gpus)
4. [Policy Evaluation & Benchmarking](#4-policy-evaluation--benchmarking)
5. [Dataset Specifications (LeRobot v3.0)](#5-dataset-specifications-lerobot-v30)

---

## 1. Docker Setup & Compilation

### Compile/Build Docker Image
```bash
./docker_run.sh --build
```

### Launch Interactive Docker Sandbox
```bash
./docker_run.sh
```

---

## 2. Data Collection

### Manual Teleoperation Mode
Control the Piper arm manually using a **3Dconnexion SpaceMouse** or **Keyboard** to record demonstration trajectories:

```bash
./docker_run.sh --task "place the red block in blue bin" --data-dir data/red_block_dataset
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
```bash
./docker_run.sh --auto-collect --num-episodes 10 --task "place the red block in blue bin" --data-dir data/red_block_dataset
```

#### Fast Headless Mode:
```bash
./docker_run.sh --auto-collect --num-episodes 50 --task "place the red block in blue bin" --data-dir data/red_block_dataset --headless
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

### Training on Local Machine
Train policies directly using official **Hugging Face LeRobot** (`lerobot-train` CLI integration) with step-based schedules, EMA, learning rate warmups, and serialized pre/postprocessors. Pass your Hugging Face dataset repo ID (`--repo-id`) and optional local dataset root (`--dataset-root`):

> 💡 **Automatic Directory Versioning**: Output directories auto-increment (`outputs/train/smolvla_piper`, `outputs/train/smolvla_piper_1`, `outputs/train/smolvla_piper_2`, etc.) so subsequent training runs never overwrite past checkpoints.

#### 1. Fine-Tune SmolVLA (Vision-Language-Action Foundation Model)
```bash
./docker_run.sh --train \
  --repo-id <HF_USER>/<DATASET_REPO_ID> \
  --dataset-root data/red_block_dataset \
  --policy-type smolvla \
  --pretrained-path lerobot/smolvla_base \
  --steps 20000
```

#### 2. Train Diffusion Policy (Recommended for Piper Arm)
```bash
# Option A: Train by defining Epochs & Batch Size
./docker_run.sh --train \
  --repo-id <HF_USER>/<DATASET_REPO_ID> \
  --dataset-root data/red_block_dataset \
  --policy-type diffusion \
  --epochs 20 \
  --batch-size 32

# Option B: Train by defining Steps directly
./docker_run.sh --train \
  --repo-id <HF_USER>/<DATASET_REPO_ID> \
  --dataset-root data/red_block_dataset \
  --policy-type diffusion \
  --steps 20000 \
  --batch-size 16
```

#### 3. Train ACT (Action Chunking with Transformers)
```bash
./docker_run.sh --train \
  --repo-id <HF_USER>/<DATASET_REPO_ID> \
  --dataset-root data/red_block_dataset \
  --policy-type act \
  --epochs 20 \
  --batch-size 16
```

---

### Configurable Training Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| **`--epochs N`** | `None` | Set training duration by **number of epochs** (automatically calculates matching steps) |
| **`--steps N`** | `20000` | Set training duration by **total steps** |
| **`--batch-size N`** | `16` | Mini-batch size for training |
| **`--save-freq N`** | `20000` | Frequency in steps to save intermediate checkpoints (e.g., `--save-freq 5000`) |
| **`--policy-type TYPE`** | `act` | Model architecture (`act`, `diffusion`, or `smolvla`) |
| **`--dataset-root PATH`** | `data/red_block_dataset` | Path to local dataset folder |
| **`--output-dir PATH`** | Auto-incremented | Custom checkpoint output folder |

---

### Step Count vs. Epoch Breakdown (16,960 Frames, Batch Size 16)

LeRobot training is **step-based**. 1 full epoch (100% of dataset) = **1,060 steps**.

| Training Steps | Frames Processed | Equivalent Epochs | Usage Recommendation |
| :--- | :--- | :--- | :--- |
| **200 steps** | 3,200 frames | **~0.19 Epochs** (~19%) | Quick code / GPU sanity check |
| **1,060 steps** | 16,960 frames | **1.0 Epoch** (100%) | Single epoch benchmark |
| **20,000 steps** *(Default)* | 320,000 frames | **~18.9 Epochs** | standard policy training |
| **50,000 steps** | 800,000 frames | **~47.2 Epochs** | Full convergence training |

---

### Training on Cloud GPUs (RunPod, Lambda, AWS, GCP) & Model Export

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

Benchmark trained policy checkpoints in closed-loop MuJoCo simulation with full preprocessor normalization and action un-normalization:

### Complete Evaluation Command
```bash
./docker_run.sh --eval \
    --checkpoint checkpoints/act_lerobot/best_model \
    --num-episodes 10 \
    --max-steps 350 \
    --headless \
    --save-video \
    --video-dir eval_videos
```

### Available Evaluation Parameters:
| Argument | Default | Description |
|---|---|---|
| `--checkpoint PATH` | `checkpoints/act_lerobot/best_model` | Model directory or Hugging Face Hub repo ID |
| `--num-episodes N` | `10` | Number of test evaluation episodes |
| `--max-steps N` | `350` | Maximum timesteps per episode before timeout |
| `--headless` | `False` | Run in headless mode without opening GUI viewer |
| `--save-video` | `False` | Save rollout video MP4s |
| `--video-dir PATH` | `eval_videos` | Output directory for rollout videos |

---

## 5. Dataset Specifications (LeRobot v3.0)

Datasets are saved in the official **LeRobot v3.0** schema with chunked Parquet tables and H.264 compressed MP4 videos:

```
data/red_block_dataset/
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
        ├── observation.images.extrinsic/file-000.mp4   # Scene Overview (H.264)
        └── observation.images.topdown/file-000.mp4     # Topdown View (H.264)
```
