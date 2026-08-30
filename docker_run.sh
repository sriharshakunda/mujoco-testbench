#!/bin/bash
# ==============================================================================
# Piper Arm MuJoCo Teleoperation & LeRobot VLA Pipeline - Docker Runner
# ==============================================================================

set -e

IMAGE_NAME="mujoco-testbench:latest"

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
if [ -d "/dev/input" ] && [ -w "/dev/input" ]; then
    chmod -R a+rw /dev/input 2>/dev/null || true
fi

# 5. Route subcommands (--auto-collect, --train, --eval, --viz, --upload, --export-video, --convert, or default app.py)
MODE="app"
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --auto-collect|--collect)
            MODE="auto_collect"
            ;;
        --train)
            MODE="train"
            ;;
        --eval|--evaluate)
            MODE="eval"
            ;;
        --lerobot-train)
            MODE="lerobot_train"
            ;;
        --lerobot-eval)
            MODE="lerobot_eval"
            ;;
        --viz)
            MODE="viz"
            ;;
        --upload)
            MODE="upload"
            ;;
        --export-video|--export)
            MODE="export_video"
            ;;
        --convert)
            MODE="convert"
            ;;
        --hil-serl)
            MODE="hil_serl"
            ;;
        --reward-classifier)
            MODE="reward_classifier"
            ;;
        --dagger)
            MODE="dagger"
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

case "$MODE" in
    auto_collect)
        CMD=(python -m src.auto_collect)
        ;;
    train)
        CMD=(python -m src.train_policy)
        ;;
    eval)
        CMD=(python -m src.evaluate_policy)
        ;;
    lerobot_train)
        CMD=(lerobot-train)
        ;;
    lerobot_eval)
        CMD=(lerobot-eval)
        ;;
    hil_serl)
        CMD=(python -m lerobot.rl.gym_manipulator)
        ;;
    reward_classifier)
        CMD=(python -m src.reward_classifier)
        ;;
    dagger)
        CMD=(lerobot-rollout --strategy.type=dagger)
        ;;
    viz)
        CMD=(python -m src.visualize_dataset)
        ;;
    upload)
        CMD=(python -m src.upload_dataset)
        ;;
    export_video)
        CMD=(python -m src.export_video)
        ;;
    convert)
        CMD=(python -m src.convert_dataset_to_mp4)
        ;;
    app)
        if [ "${ARGS[0]}" == "python" ] || [ "${ARGS[0]}" == "bash" ]; then
            CMD=()
        else
            CMD=(python app.py)
        fi
        ;;
esac

echo -e "\033[1;32m[Docker] Launching container (Command: ${CMD[*]} ${ARGS[*]}) ...\033[0m"

HF_ENV_FLAG=()
if [ -n "$HF_TOKEN" ]; then
    HF_ENV_FLAG=(-e HF_TOKEN="$HF_TOKEN")
fi

# 6. Run container interactively
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    $GPU_FLAG \
    "${HF_ENV_FLAG[@]}" \
    --net=host \
    --ipc=host \
    --privileged \
    -e USER="$(whoami)" \
    -e LOGNAME="$(whoami)" \
    -e HOME="/tmp" \
    -e TORCH_HOME="/tmp/torch_cache" \
    -e DISPLAY="$DISPLAY" \
    -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
    -e PYTHONPATH="/app" \
    -e HF_HOME="/tmp/.cache/huggingface" \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/input:/dev/input:rw \
    -v "$(pwd)":/app \
    -v "$HOME/.cache/huggingface:/tmp/.cache/huggingface:rw" \
    -v "$HOME/.cache/torch:/tmp/torch_cache:rw" \
    -w /app \
    "$IMAGE_NAME" \
    "${CMD[@]}" "${ARGS[@]}"

