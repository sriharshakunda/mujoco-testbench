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

```bash
docker build -t mujoco-testbench .
docker run -it mujoco-testbench
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

