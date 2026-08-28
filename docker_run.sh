#!/bin/bash
# ==============================================================================
# Piper Arm MuJoCo Teleoperation & LeRobot VLA Pipeline - Docker Runner
# ==============================================================================

set -e

IMAGE_NAME="piper_robot_decompose:latest"

# 1. Allow local X11 connections for GUI viewer
xhost +local:root 2>/dev/null || true
xhost +local:$(whoami) 2>/dev/null || true

# 2. Build Docker Image (if requested with --build)
if [ "$1" == "--build" ]; then
    echo -e "\033[1;34m[Docker] Rebuilding image: $IMAGE_NAME ...\033[0m"
    docker build -t "$IMAGE_NAME" -f Dockerfile .
    shift
fi

# 3. Check GPU support
GPU_FLAG=""
if command -v nvidia-smi &>/dev/null; then
    GPU_FLAG="--gpus all"
fi

# 4. Check /dev/input permissions for SpaceMouse
if [ -d "/dev/input" ]; then
    sudo chmod -R a+rw /dev/input 2>/dev/null || true
fi

# 5. Route subcommands (--viz, --upload, --export-video, or default app.py)
CMD=(python app.py)
if [ "$1" == "--viz" ]; then
    shift
    CMD=(python -m src.visualize_dataset)
elif [ "$1" == "--upload" ]; then
    shift
    CMD=(python -m src.upload_dataset)
elif [ "$1" == "--export-video" ] || [ "$1" == "--export" ]; then
    shift
    CMD=(python -m src.export_video)
elif [ "$1" == "python" ] || [ "$1" == "bash" ]; then
    CMD=()
fi

echo -e "\033[1;32m[Docker] Launching container (Command: ${CMD[*]} $@) ...\033[0m"

# 6. Run container interactively
docker run -it --rm \
    $GPU_FLAG \
    --net=host \
    --ipc=host \
    --privileged \
    -e DISPLAY="$DISPLAY" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/input:/dev/input:rw \
    -v "$(pwd)":/app \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw" \
    -w /app \
    "$IMAGE_NAME" \
    "${CMD[@]}" "$@"
