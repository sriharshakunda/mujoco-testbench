# Robot Decompose - Basic MuJoCo Setup

A minimal MuJoCo simulation environment for robot learning.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Verification

```bash
python -c "import mujoco; print('MuJoCo installed successfully')"
```

## Docker Setup

### Build

```bash
docker build -t mujoco-testbench .
```

### Run — Teleoperation Simulation

Requires a SpaceMouse and an X11 display. Launches the full Piper arm teleop pipeline with live camera views and telemetry.

```bash
# Using the convenience script (handles X11 forwarding, GPU, and /dev/input permissions)
./docker_run.sh

# Or manually
docker run -it --rm \
    --net=host --ipc=host --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/input:/dev/input:rw \
    -v $(pwd):/app \
    mujoco-testbench python app.py
```

#### Configurable Parameters

All parameters are appended after `./docker_run.sh` or after `python app.py` in the manual command.

| Parameter | Default | Description |
|---|---|---|
| `--task` | `"pick up the red cube and place it into the bin"` | Language instruction written into each recorded episode for VLA training |
| `--data-dir` | `data/lerobot_dataset` | Directory where recorded episodes are saved |
| `--target X Y Z` | `0.42 0.22 0.31` | Goal marker position in the simulation (metres) |
| `--exposure` | `1.0` | Camera brightness multiplier (e.g. `1.3` to brighten) |
| `--no-camera` | _(camera on by default)_ | Disable the live multi-camera visualizer window |

Episodes are numbered automatically (`episode_000000`, `episode_000001`, …) inside `--data-dir`. Each new run appends to the existing dataset.

**Examples:**

```bash
# Custom task label and save location
./docker_run.sh --task "place the blue block on the shelf" --data-dir data/blue_block_dataset

# Move the goal marker and dim the camera
./docker_run.sh --target 0.35 0.10 0.25 --exposure 0.8

# Run without the camera window (faster, headless-friendly)
./docker_run.sh --no-camera
```

### Run — Visualization Only

Replay and inspect a recorded LeRobot dataset without hardware.

```bash
# Using the convenience script
./docker_run.sh --viz

# Or manually
docker run -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/app \
    mujoco-testbench python -m src.visualize_dataset
```

## Project Structure

```
src/
├── environment/     # MuJoCo environment definitions
├── utils/          # Utility functions
└── __init__.py
```

## Requirements

- Python 3.10+
- MuJoCo 2.3.0+
- NumPy, SciPy, Matplotlib

