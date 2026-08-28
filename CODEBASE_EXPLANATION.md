# MuJoCo Testbench Codebase Explanation

## 📁 Project Structure Overview

```
mujoco-testbench/

├── app.py                          # Main teleoperation application
├── snapshot.py                     # Camera capture and rendering
├── make_video.py                   # Video recording utility
├── assets/
│   └── piper.xml                   # MuJoCo robot model (MJCF format)
├── src/
│   ├── environment/
│   │   ├── env.py                  # MuJoCo environment wrapper
│   │   └── tasks.py                # Task definitions (reaching, etc.)
│   ├── controllers/
│   │   ├── ik_controller.py        # Inverse kinematics solver
│   │   └── pinocchio_controller.py # Pinocchio-based kinematics
│   ├── spacemouse.py               # SpaceMouse input device handling
│   ├── camera.py                   # Wrist camera interface
│   ├── lerobot_dataset.py          # LeRobot dataset utilities
│   ├── teleop_tui.py               # Terminal UI for teleoperation
│   └── utils/
│       ├── config.py               # Configuration management
│       └── logging_utils.py        # Logging utilities
├── scripts/
│   ├── train_il.py                 # Imitation learning training
│   ├── train_rl.py                 # Reinforcement learning training
│   ├── evaluate_policy.py          # Policy evaluation
│   └── main.py                     # Main orchestration script
└── tests/                          # Unit tests for all modules
```

---

## 🤖 Core Components Explained

### 1. **src/environment/env.py** - MuJoCo Environment

**Purpose:** Wraps the MuJoCo physics simulator for the Piper arm robot.

**Key Classes & Functions:**

#### Class: `PiperEnv`
```python
PiperEnv(render_mode=None, dt=0.002, max_episode_steps=500, target_pos=None)
```

**Constructor Parameters:**
- `render_mode`: "human" (viewer) | "rgb_array" (image output) | None
- `dt`: Physics timestep (0.002s = 500 Hz)
- `max_episode_steps`: Maximum steps before episode ends
- `target_pos`: Initial target position for reaching tasks

**Important Constants:**
```python
HOME_QPOS = [0.0, -3.14, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0]
#            j1   j2    j3     j4   j5    j6    g1   g2
# j1-j6: arm joints (radians)
# g1-g2: gripper joints (radians)
```

**Key Methods:**

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `reset()` | Initialize environment to home pose | None | `(observation, info_dict)` |
| `step(action)` | Execute one physics step | `action` (7D array) | `(obs, reward, terminated, truncated, info)` |
| `set_target(pos)` | Move goal marker in sim | `pos` (3D position) | None |
| `get_ee_pos()` | Get end-effector position | None | 3D position (x, y, z) |
| `get_target_pos()` | Get target marker position | None | 3D position |
| `render()` | Capture image from camera | None | RGB image (H, W, 3) or None |
| `close()` | Cleanup resources | None | None |

**How it works:**
1. **Initialization**: Loads `piper.xml` MuJoCo model, creates physics simulation
2. **Reset**: Sets joint positions to HOME_QPOS, runs forward kinematics
3. **Step**: Applies control inputs → advances physics 1 timestep → computes reward
4. **Reward**: Based on distance to target (negative reward = how far from goal)

---

### 2. **src/controllers/ik_controller.py** - Inverse Kinematics

**Purpose:** Convert end-effector (TCP) position/orientation goals into joint commands.

**Key Helper Functions:**

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `mat2quat(rot_mat)` | Rotation matrix → quaternion | 3×3 rotation matrix | 4D quaternion [w,x,y,z] |
| `quat2mat(quat)` | Quaternion → rotation matrix | 4D quaternion | 3×3 rotation matrix |
| `euler2mat(r,p,y)` | Euler angles → rotation matrix | roll, pitch, yaw (radians) | 3×3 rotation matrix |
| `mat2euler(R)` | Rotation matrix → Euler angles | 3×3 rotation matrix | (roll, pitch, yaw) |

**Class: `DifferentialIKController`**
```python
DifferentialIKController(
    model,
    site_name="ee",           # End-effector site name
    num_arm_joints=6,         # Number of arm joints
    max_iters=50,            # Max iterations for IK solver
    tol_pos=1e-4,            # Position tolerance (0.1 mm)
    tol_rot=1e-3,            # Rotation tolerance (0.05°)
    home_qpos=None,
    use_pinocchio=False      # Use Pinocchio library if True
)
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `solve(target_pos, target_rot)` | Solve IK for target pose |
| `solve_from_tcp_delta(delta_pos, delta_rot)` | Solve IK from TCP-frame velocity |
| `step_cartesian_delta(delta_pos, delta_rot)` | Single IK step in world frame |
| `step_tcp_delta(delta_pos, delta_rot)` | Single IK step in TCP-local frame |

**How IK works:**
1. **Input**: Desired end-effector position/orientation
2. **Algorithm**: Levenberg-Marquardt (numerical IK solver)
3. **Process**: Iteratively adjusts joint angles to reach target
4. **Output**: Joint angles (qpos) that achieve target pose

---

### 3. **app.py** - Main Teleoperation Application

**Purpose:** Interactive real-time control of the Piper arm via keyboard and SpaceMouse.

**Key Components:**

#### Class: `RawKeyboard`
- **Purpose**: Non-blocking keyboard input reader
- **Methods**:
  - `__enter__()`: Enter context, set terminal to cbreak mode
  - `__exit__()`: Cleanup terminal settings
  - `read()`: Read one keystroke (non-blocking)

**Main Control Loop:**
```python
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while not should_exit:
        # 1. Read inputs (keyboard + spacemouse)
        key = keyboard.read()
        spacemouse_6dof = spacemouse.read()
        
        # 2. Process inputs → joint commands
        action = ik_controller.step_tcp_delta(...)
        
        # 3. Execute in physics
        env.step(action)
        
        # 4. Update viewer
        viewer.sync()
```

**Keyboard Controls:**

| Keys | Function |
|------|----------|
| W/S | Move TCP forward/backward |
| A/D | Move TCP left/right |
| R/F | Move TCP up/down |
| U/O | Roll (twist around Z axis) |
| I/K | Pitch (nod around X axis) |
| J/L | Yaw (pan around Y axis) |
| [ / ] | Close/open gripper |
| 1/2/3 | Speed mode (fine/normal/fast) |
| H | Return to home pose |
| P | Toggle SpaceMouse |
| Q/ESC | Quit |

**SpaceMouse Controls:**
- **Pan/Tilt/Push**: 6-DOF TCP motion (position + rotation)
- **Left Button**: Toggle gripper
- **Right Button**: Home pose

**Speed Modes:**
```python
speed_scales = {
    '1': 0.02,   # Fine (2 cm/step)
    '2': 0.05,   # Normal (5 cm/step)
    '3': 0.10    # Fast (10 cm/step)
}
```

---

### 4. **src/spacemouse.py** - SpaceMouse Input Handler

**Purpose:** Read input from 3Dconnexion SpaceMouse device.

**Class: `SpaceMouse`**
```python
SpaceMouse()
```

**Key Methods:**

| Method | Purpose | Output |
|--------|---------|--------|
| `read()` | Read current state | `{pos, rot, buttons}` |
| `get_motion()` | Get 6-DOF motion vector | 6D array [dx, dy, dz, rx, ry, rz] |
| `get_button()` | Check button press | Boolean |
| `close()` | Close device | None |

**Output Format:**
```python
{
    'pos': [dx, dy, dz],              # Translation (m)
    'rot': [roll_delta, pitch_delta, yaw_delta],  # Rotation (rad)
    'button_left': bool,
    'button_right': bool
}
```

---

### 5. **snapshot.py** - Camera Capture & Rendering

**Purpose:** Capture images from wrist camera and scene camera.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `settle_spheres()` | Drop objects into bin (physics) |
| `render_scene_cam()` | Capture overview camera image |
| `render_topdown_cam()` | Capture top-down camera |
| `render_wrist_cams()` | Capture wrist RGB + depth |
| `make_figure()` | Create 5-panel visualization |
| `save_snapshot()` | Save all camera images to disk |

**Camera Offsets:**
- **Scene Cam**: Positioned at (0.30, -0.30, 2.10) looking at workspace
- **Top-down Cam**: Directly above bin at (0.30, 0.38, 1.90)
- **Wrist Cameras**: Mounted on gripper_base, see what gripper sees

---

### 6. **src/lerobot_dataset.py** - Dataset Management

**Purpose:** Record and manage teleoperation trajectories in LeRobot format.

**Key Methods:**
- `record_episode()`: Capture one teleoperation session
- `save_to_hdf5()`: Save trajectory in LeRobot format
- `load_trajectory()`: Load for training

**Data Stored:**
```python
{
    'observations': {
        'qpos': joint_positions,
        'qvel': joint_velocities,
        'image_wrist': wrist_rgb_images
    },
    'actions': joint_commands,
    'metadata': {
        'episode_id': int,
        'timestamp': float,
        'task': str
    }
}
```

---

### 7. **Training Pipeline**

#### scripts/train_il.py - Imitation Learning
**What it does:**
1. Load recorded trajectories from LeRobot dataset
2. Train behavioral cloning model (MLP policy)
3. Loss function: MSE between predicted and actual joint commands
4. Save best checkpoint

#### scripts/train_rl.py - Reinforcement Learning
**What it does:**
1. Load pre-trained IL model (optional warmstart)
2. Train with PPO or SAC algorithm
3. Reward: distance to target
4. Improve beyond demonstration quality

#### scripts/evaluate_policy.py - Policy Evaluation
**What it does:**
1. Load trained policy checkpoint
2. Execute in simulation
3. Compute success rate, trajectory smoothness, etc.
4. Generate report

---

## 🔄 Typical Workflow

### 1. **Teleoperation (Data Collection)**
```bash
python app.py
```
- Operator controls arm with keyboard/SpaceMouse
- Collects trajectories in LeRobot format
- Saves to `data/` directory

### 2. **Train IL Model**
```bash
python scripts/train_il.py --dataset data/piper_demo_v1.hdf5
```
- Trains neural network to mimic operator
- Output: `checkpoints/il_model.pt`

### 3. **Evaluate IL**
```bash
python scripts/evaluate_policy.py --checkpoint checkpoints/il_model.pt
```
- Tests how well model reproduces demonstrations
- Metrics: success rate, trajectory error

### 4. **Train RL (Optional)**
```bash
python scripts/train_rl.py --il-checkpoint checkpoints/il_model.pt
```
- Uses IL model as starting point
- Improves via reinforcement learning
- Output: `checkpoints/rl_model.pt`

---

## 🧬 Key Concepts

### **MuJoCo (piper.xml)**
- **Model**: Robot description (links, joints, actuators)
- **Data**: Current state (positions, velocities, contacts)
- **Forward Kinematics**: Joint angles → end-effector position
- **Inverse Kinematics**: End-effector goal → joint commands

### **TCP Frame (Tool Center Point)**
- The end-effector's local coordinate frame
- W/S/A/D/R/F move TCP relative to itself (not world)
- Intuitive for teleoperation (like controlling a camera)

### **Quaternion vs Euler**
- **Quaternion**: [w, x, y, z] - smooth interpolation, no gimbal lock
- **Euler**: [roll, pitch, yaw] - human-readable angles

### **Reward Function**
```python
reward = -distance_to_target
# reward > -0.02 means within 2cm (success!)
```

---

## 🎯 State & Action Spaces

### **Observation (16D)**
```python
obs = {
    'qpos': [j1, j2, j3, j4, j5, j6, g1, g2],    # 8D: joint positions
    'qvel': [v1, v2, v3, v4, v5, v6, vg1, vg2],  # 8D: joint velocities
}
```

### **Action (7D)**
```python
action = [a1, a2, a3, a4, a5, a6, a_gripper]
# Each element in [-1.0, 1.0] (normalized)
# Actuator converts to joint torques
```

### **TCP Delta (6D)**
```python
tcp_delta = [dx, dy, dz, droll, dpitch, dyaw]
# Used for keyboard/spacemouse control
# IK solver converts to joint commands
```

---

## 🔧 Important Files to Modify

| File | What to Change |
|------|-----------------|
| `assets/piper.xml` | Robot model (if URDF changes) |
| `src/environment/env.py` | Observation/reward logic |
| `src/controllers/ik_controller.py` | IK solver parameters |
| `app.py` | Control mappings, speed scales |
| `scripts/train_il.py` | Training hyperparameters |

---

## 📊 Data Flow Diagram

```
SpaceMouse/Keyboard
        ↓
    app.py (reads input)
        ↓
IKController (solves inverse kinematics)
        ↓
    action = joint commands [7D]
        ↓
PiperEnv.step(action)
        ├→ Physics simulation (mujoco.mj_step)
        ├→ Forward kinematics
        └→ Reward computation
        ↓
    obs [16D], reward [1D], done [bool]
        ↓
Viewer / Camera / LeRobot Dataset
```

---

## 🚀 Getting Started

1. **Understand the environment**: Read `env.py` and understand MuJoCo basics
2. **Try teleoperation**: `python app.py` and control the arm
3. **Check camera views**: `python snapshot.py` to see all camera angles
4. **Collect data**: Use app.py to record demonstrations
5. **Train model**: Use `scripts/train_il.py` to learn from data

---

## 📚 Additional Resources

- **MuJoCo Docs**: https://mujoco.readthedocs.io/
- **Pinocchio**: Rigid-body dynamics library (used for analytical IK)
- **LeRobot**: Hugging Face robotics dataset format
- **MJCF Format**: MuJoCo's XML robot description language

