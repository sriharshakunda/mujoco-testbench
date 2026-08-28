#!/bin/bash
# ==============================================================================
# Setup Python Virtual Environment for LeRobot & Piper MuJoCo Testbench
# ==============================================================================

set -e

VENV_DIR="venv"

echo "===================================================================="
echo "      Setting up Python Virtual Environment in ./${VENV_DIR}"
echo "===================================================================="

# 1. Create Virtual Environment if it doesn't exist or is invalid
if [ ! -f "${VENV_DIR}/bin/python3" ] && [ ! -f "${VENV_DIR}/bin/python" ]; then
    echo "[VENV] Creating virtual environment with $(python3 --version)..."
    rm -rf "$VENV_DIR"
    python3 -m venv --copies "$VENV_DIR"
    echo "[VENV] Created ${VENV_DIR} successfully."
else
    echo "[VENV] Virtual environment ./${VENV_DIR} already exists."
    # Clean up any leftover pip temporary directories (e.g., ~umpy)
    find "${VENV_DIR}" -name "~*" -type d -exec rm -rf {} + 2>/dev/null || true
fi

# Locate Python binary in venv
if [ -f "${VENV_DIR}/bin/python3" ]; then
    VENV_PYTHON="${VENV_DIR}/bin/python3"
else
    VENV_PYTHON="${VENV_DIR}/bin/python"
fi

# 2. Upgrade Core Packaging Tools
echo "[VENV] Upgrading pip, setuptools, and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 3. Install Dependencies
echo "[VENV] Installing PyTorch, LeRobot (from HuggingFace official GitHub repo), transformers, datasets, accelerate, num2words, and project dependencies (this may take 2-3 mins)..."
"$VENV_PYTHON" -m pip install torch torchvision numpy datasets draccus einops deepdiff transformers "diffusers>=0.30.0" "huggingface-hub>=0.25.0" accelerate num2words "av==12.3.0" pandas pyarrow imageio opencv-python matplotlib scipy mujoco
"$VENV_PYTHON" -m pip install --upgrade "git+https://github.com/huggingface/lerobot.git"

echo "===================================================================="
echo "✓ Virtual environment setup complete!"
echo "To activate in your terminal run:"
echo "    source ${VENV_DIR}/bin/activate"
echo "===================================================================="
