# Agilex Piper 6-DOF Robotic Arm MuJoCo Testbench & LeRobot VLA Pipeline

A high-fidelity simulation environment, autonomous multi-modal demonstration collector, and imitation learning benchmark for the **Agilex Piper 6-DOF manipulator** equipped with an analog parallel-jaw gripper.

---

## Table of Contents
1. [Quick Start & Docker Setup](#quick-start--docker-setup)
2. [Automated Data Collection (Recommended)](#1-automated-data-collection)
3. [Manual Teleoperation & SpaceMouse Control](#2-manual-teleoperation)
4. [Dataset Inspection & Video Export](#3-dataset-inspection--video-export)
5. [Hugging Face Hub Upload & Online Visualizer](#4-hugging-face-hub-upload)
6. [Policy Training (ACT & SmolVLA)](#5-policy-training)
7. [Policy Evaluation & Benchmark](#6-policy-evaluation--benchmark)
8. [Dataset Specification (LeRobot v2.0)](#dataset-specification)

---

## Quick Start & Docker Setup

### 1. Build Container Image
```bash
./docker_run.sh --build
```

### 2. Verify Simulation & Kinematics
```bash
./docker_run.sh python -c "import mujoco; print('MuJoCo ready!')"
```

---

## 1. Automated Data Collection

Collect verified multi-modal demonstration episodes automatically without human teleoperation. The autonomous controller generates realistic trajectories with tabletop cube pose randomization, grasping the red cube and placing it into the target blue bin.

### Interactive Live GUI
Watch the arm autonomously pick and place each episode:
```bash
./docker_run.sh --auto-collect --num-episodes 10 --task "place the red block in blue bin" --data-dir data/red_block_dataset
```

### Fast Headless Mode
Generate large batches of demonstration data in the background at maximum simulation speed:
```bash
./docker_run.sh --auto-collect --num-episodes 50 --task "place the red block in blue bin" --data-dir data/red_block_dataset --headless
```

#### Parameters:
| Argument | Default | Description |
|---|---|---|
| `--auto-collect` | - | Triggers autonomous pick-and-place collection |
| `--num-episodes N` | `10` | Number of verified successful episodes to record |
| `--task "PROMPT"` | `"pick up the red cube and place it into the blue bin"` | Language instruction prompt |
| `--data-dir PATH` | `data/red_block_dataset` | Directory to store dataset |
| `--fps FPS` | `30` | Trajectory capture frame rate |
| `--headless` | `False` | Run headless without opening X11 GUI window |

---

## 2. Manual Teleoperation

Control the arm manually using a **3Dconnexion SpaceMouse** or **Keyboard**:

```bash
./docker_run.sh --task "place the red block in blue bin" --data-dir data/red_block_dataset
```

### Keyboard Controls:
| Key | Action |
|---|---|
| `W / S` | Forward / Backward along table workspace ($+X / -X$) |
| `A / D` | Left / Right across table workspace ($+Y / -Y$) |
| `R / F` | Elevation Height ($+Z / -Z$) |
| `U / O` | Wrist Roll $\pm$ |
| `I / K` | Wrist Pitch $\pm$ |
| `J / L` | Wrist Yaw $\pm$ |
| `[ / ]` | Open / Close Gripper |
| `Space` or `C` | Start / Stop & Save Episode Recording |
| `N` | Discard Current Episode Buffer |
| `H` | Reset to Home Pose & Re-randomize Cubes |
| `Q` / `Esc` | Quit Teleoperation |

---

## 3. Dataset Inspection & Video Export

### Visualizer Window
Replay recorded episodes with synchronized 4-camera views:
```bash
./docker_run.sh --viz --data-dir data/red_block_dataset --episode 0
```

### Export Multi-View HD Video
Export a stitched 4-camera composite HD video (`1280x1440` @ 30 FPS) with telemetry overlays:
```bash
./docker_run.sh --export-video --data-dir data/red_block_dataset --episode 0 --output episode_0000.mp4
```

---

## 4. Hugging Face Hub Upload

The dataset is recorded in official **LeRobot v2.0 / v2.1** format (`.parquet` tables, chunked `.mp4` videos, `meta/stats.json`), allowing instant visualization on the [Hugging Face LeRobot Visualizer Space](https://huggingface.co/spaces/lerobot/visualize_dataset).

### 1. Authenticate with Hugging Face (Once)
```bash
pip install huggingface_hub
huggingface-cli login
```

### 2. Upload Dataset
```bash
./docker_run.sh --upload --data-dir data/red_block_dataset --repo-id your_hf_username/dataset_name
```
*Add `--private` to create a private repository.*

---

## 5. Policy Training with Hugging Face LeRobot

Train state-of-the-art imitation learning and Vision-Language-Action (VLA) foundation models directly using the official **[Hugging Face LeRobot](https://github.com/huggingface/lerobot)** library.

### 1. Fine-Tune SmolVLA (Vision-Language-Action Foundation Model)
SmolVLA is initialized from official pretrained base weights (`lerobot/smolvla_base` / `SmolVLM2-500M-Video-Instruct`) and fine-tunes the action flow-matching expert head conditioned on your dataset's natural language instructions (`"place the red block in blue bin"`) and multi-camera feeds.

```bash
# Fine-tune SmolVLA from pretrained base weights on GPU
./docker_run.sh --train --dataset-dir data/red_block_dataset --policy-type smolvla --pretrained-path lerobot/smolvla_base --epochs 5 --batch-size 8 --output-dir checkpoints/smolvla_lerobot
```

### 2. Train ACT (Action Chunking with Transformers)
ACT uses ResNet-18 vision encoders + CVAE + Transformer Decoders for fast, millimeter-accurate manipulation trajectories.

```bash
# Train ACT policy for 30 epochs on GPU
./docker_run.sh --train --dataset-dir data/red_block_dataset --policy-type act --epochs 30 --batch-size 16 --chunk-size 30 --output-dir checkpoints/act_lerobot
```

### 3. Train Diffusion Policy
Diffusion Policy uses score-based diffusion denoising to represent multi-modal trajectory distributions.

```bash
# Train Diffusion Policy on GPU
./docker_run.sh --train --dataset-dir data/red_block_dataset --policy-type diffusion --epochs 30 --batch-size 16 --output-dir checkpoints/diffusion_lerobot
```

#### Training Parameters:
| Argument | Default | Description |
|---|---|---|
| `--dataset-dir` | `data/red_block_dataset` | Path to recorded LeRobot dataset |
| `--policy-type` | `act` | Model architecture: `smolvla`, `act`, or `diffusion` |
| `--pretrained-path` | `None` (or `lerobot/smolvla_base`) | Pretrained model checkpoint / Hub repo to fine-tune from |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `16` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate with cosine schedule |
| `--chunk-size` | `30` | Action prediction chunk horizon (1.0s @ 30 FPS) |
| `--output-dir` | `checkpoints/act_lerobot` | Output model folder (`best_model/model.safetensors`, `config.json`) |

---

## 6. Policy Evaluation & Benchmark

Benchmark your trained policy checkpoint in closed-loop MuJoCo simulation across randomized test episodes:

### Evaluate SmolVLA
```bash
./docker_run.sh --eval --checkpoint checkpoints/smolvla_lerobot/best_model --num-episodes 10
```

### Evaluate ACT
```bash
./docker_run.sh --eval --checkpoint checkpoints/act_lerobot/best_model --num-episodes 10
```

### Headless Benchmark & Export Rollout Videos
```bash
./docker_run.sh --eval --checkpoint checkpoints/smolvla_lerobot/best_model --num-episodes 20 --headless --save-video --video-dir eval_videos
```

#### Evaluation Metrics Output:
- **Success Rate (%)**: Percentage of episodes where the cube was placed inside the target blue bin.
- **Execution Length**: Average timesteps / seconds to completion.
- **Rollout Videos**: Saved to `eval_videos/eval_episode_XXXX.mp4`.

---

## 7. Human-in-the-Loop Reinforcement Learning (HIL-SERL)

**HIL-SERL** (*Human-in-the-Loop Sample-Efficient Robotic Reinforcement Learning*, developed by UC Berkeley RAIL) enables dexterous manipulation by combining off-policy RL (SAC / DrQ-v2) with live human interventions and vision-based reward classifiers.

### Key Architecture:
1. **Offline Demonstrations**: Initialize actor-critic replay buffers and train a binary success classifier on human teleoperated episodes.
2. **Actor-Learner Interaction Loop**: The RL policy acts in the MuJoCo simulation or real arm while an asynchronous learner updates Q-networks and policy weights.
3. **Live Human Interventions**: Whenever the robot enters an unproductive or unsafe state, the human takes control via SpaceMouse or Keyboard (`W/A/S/D/R/F`). The intervention transitions are stored directly in the replay buffer with high priority, guiding the policy toward success in <1–2 hours of interaction.

### Running HIL-SERL with Piper:
```bash
# 1. Collect 10 baseline demonstrations for the reward classifier
./docker_run.sh --task "place the red block in blue bin" --data-dir data/hil_serl_demos

# 2. Launch HIL-SERL actor loop with live SpaceMouse intervention
./docker_run.sh python -m src.environment.hil_serl_agent --demo-dir data/hil_serl_demos --interactive
```

---

## Dataset Specification

```
data/red_block_dataset/
├── meta/
│   ├── info.json              # Schema, codebase version v2.0, split definitions
│   ├── stats.json             # Normalization statistics (min, max, mean, std)
│   ├── episodes.jsonl         # Episode index, lengths, and task mapping
│   └── tasks.jsonl            # Language task prompts
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # [observation.state, action, timestamp, indices]
│       └── episode_000000.npz      # Fast NumPy replay cache
└── videos/
    └── chunk-000/
        ├── observation.images.wrist/episode_000000.mp4       # (240x320) Wrist RGB
        ├── observation.images.wrist_depth/episode_000000.mp4 # (240x320) Turbo Colormap Depth
        ├── observation.images.extrinsic/episode_000000.mp4   # (240x320) Scene Overview
        └── observation.images.topdown/episode_000000.mp4     # (240x320) Topdown View
```

