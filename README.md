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

