FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for MuJoCo + GLFW/X11 viewer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    pkg-config \
    libopenblas-dev \
    # X11 / GLFW runtime (required for mujoco viewer)
    libx11-6 \
    libxrender1 \
    libxext6 \
    libxkbcommon0 \
    libgl1 \
    libglx-mesa0 \
    libglfw3 \
    libosmesa6 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    python3-tk \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create necessary directories
RUN mkdir -p /data /models /logs

# Default command
CMD ["bash"]
