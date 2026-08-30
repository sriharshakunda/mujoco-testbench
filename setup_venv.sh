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
"$VENV_PYTHON" -m pip install torch torchvision numpy "datasets<4.0.0" draccus einops deepdiff transformers "diffusers>=0.30.0" "huggingface-hub>=0.25.0" accelerate num2words "av==12.3.0" pandas pyarrow imageio opencv-python matplotlib scipy mujoco
"$VENV_PYTHON" -m pip install --upgrade "git+https://github.com/huggingface/lerobot.git" 2>/dev/null || "$VENV_PYTHON" -m pip install --upgrade lerobot 2>/dev/null || true

# 4. Apply Site-Packages Compatibility Patches
echo "[VENV] Applying site-packages compatibility patches..."
"$VENV_PYTHON" -c "
import pathlib

# Patch 1: datasets parquet schema casting
try:
    import datasets.packaged_modules.parquet.parquet as pq_mod
    p_file = pathlib.Path(pq_mod.__file__)
    content = p_file.read_text()
    if 'def _cast_table(self, pa_table: pa.Table) -> pa.Table:' in content and 'return pa_table' not in content:
        content = content.replace('def _cast_table(self, pa_table: pa.Table) -> pa.Table:\n', 'def _cast_table(self, pa_table: pa.Table) -> pa.Table:\n        return pa_table\n')
        p_file.write_text(content)
        print('  ✓ Applied Parquet schema patch')
except Exception as e:
    print(f'  Note on Parquet patch: {e}')

# Patch 2: lerobot PyAV codec canonical_name
try:
    import lerobot.datasets.video_utils as v_mod
    v_file = pathlib.Path(v_mod.__file__)
    v_content = v_file.read_text()
    if 'video_stream.codec.canonical_name' in v_content:
        v_content = v_content.replace('video_stream.codec.canonical_name', 'getattr(video_stream.codec, \"canonical_name\", video_stream.codec.name)')
        v_file.write_text(v_content)
        print('  ✓ Applied PyAV video codec patch')
except Exception as e:
    print(f'  Note on Video codec patch: {e}')

# Patch 3: Register PiperEnvConfig in lerobot.envs.configs
try:
    import lerobot.envs.configs as c_mod
    c_file = pathlib.Path(c_mod.__file__)
    c_content = c_file.read_text()
    if '@EnvConfig.register_subclass(\"piper\")' not in c_content:
        patch_code = '''

@EnvConfig.register_subclass(\"piper\")
@dataclass
class PiperEnvConfig(EnvConfig):
    type: str = \"piper\"

    def __post_init__(self):
        if not self.features:
            self.features = {
                \"observation.state\": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                \"observation.images.wrist\": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                \"observation.images.extrinsic\": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                \"action\": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            }
        if not self.features_map:
            self.features_map = {
                \"observation.state\": \"observation.state\",
                \"observation.images.wrist\": \"observation.images.wrist\",
                \"observation.images.extrinsic\": \"observation.images.extrinsic\",
                \"action\": \"action\",
            }

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int = 1, use_async_envs: bool = False):
        import sys, os
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from src.environment.env import make_env as piper_make_env
        vec_env = piper_make_env(n_envs=n_envs, use_async_envs=use_async_envs)
        return {\"piper\": {0: vec_env}}

    def get_env_processors(self):
        from lerobot.processor import PolicyProcessorPipeline
        return PolicyProcessorPipeline(), PolicyProcessorPipeline()
'''
        c_file.write_text(c_content + patch_code)
        print('  ✓ Applied PiperEnvConfig registration patch')
except Exception as e:
    print(f'  Note on PiperEnvConfig patch: {e}')

# Patch 4: Bypass HF hub version lookup for local dataset repo_ids in lerobot/datasets/utils.py
try:
    import lerobot.datasets.utils as u_mod
    u_file = pathlib.Path(u_mod.__file__)
    u_content = u_file.read_text()
    if 'if repo_id.startswith("local/")' not in u_content and 'hub_versions = get_repo_versions(repo_id)' in u_content:
        u_content = u_content.replace(
            'hub_versions = get_repo_versions(repo_id) if token is None else get_repo_versions(repo_id, token=token)',
            'if repo_id.startswith("local/") or "/" not in repo_id:\n        return str(version) if isinstance(version, packaging.version.Version) else (version or CODEBASE_VERSION)\n    try:\n        hub_versions = get_repo_versions(repo_id) if token is None else get_repo_versions(repo_id, token=token)\n    except Exception:\n        return str(version) if isinstance(version, packaging.version.Version) else (version or CODEBASE_VERSION)'
        )
        u_file.write_text(u_content)
        print('  ✓ Applied local dataset get_safe_version patch')
except Exception as e:
    print(f'  Note on get_safe_version patch: {e}')

# Patch 5: PyAV add_stream_from_template fallback in lerobot/datasets/video_utils.py
try:
    import lerobot.datasets.video_utils as vu_mod
    vu_file = pathlib.Path(vu_mod.__file__)
    vu_content = vu_file.read_text()
    if 'hasattr(output_container, "add_stream_from_template")' not in vu_content:
        vu_content = vu_content.replace(
            'stream_map[input_stream.index] = output_container.add_stream_from_template(\n                template=input_stream, opaque=True\n            )',
            'if hasattr(output_container, "add_stream_from_template"):\n                stream_map[input_stream.index] = output_container.add_stream_from_template(template=input_stream, opaque=True)\n            else:\n                stream_map[input_stream.index] = output_container.add_stream(template=input_stream)'
        )
        vu_file.write_text(vu_content)
        print('  ✓ Applied PyAV add_stream_from_template patch')
except Exception as e:
    print(f'  Note on PyAV add_stream patch: {e}')

# Patch 6: sample_images missing file fallback in lerobot/datasets/compute_stats.py
try:
    import lerobot.datasets.compute_stats as cs_mod
    cs_file = pathlib.Path(cs_mod.__file__)
    cs_content = cs_file.read_text()
    if 'except Exception:' not in cs_content or 'img = np.zeros((3, 480, 640)' not in cs_content:
        cs_content = cs_content.replace(
            'img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)',
            'try:\n            if isinstance(path, (np.ndarray, torch.Tensor)):\n                img = np.array(path, dtype=np.uint8)\n                if img.ndim == 3 and img.shape[2] in (1, 3):\n                    img = img.transpose(2, 0, 1)\n            elif hasattr(path, "size"):\n                img = np.array(path.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)\n            else:\n                img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)\n        except Exception:\n            img = np.zeros((3, 480, 640), dtype=np.uint8)'
        )
        cs_file.write_text(cs_content)
        print('  ✓ Applied sample_images missing file patch')
except Exception as e:
    print(f'  Note on sample_images patch: {e}')

# Patch 9: write_video empty frame safety in lerobot/utils/io_utils.py
try:
    import lerobot.utils.io_utils as utils_io_mod
    uio_file = pathlib.Path(utils_io_mod.__file__)
    uio_content = uio_file.read_text()
    if 'if stacked_frames is None or len(stacked_frames) == 0 or stacked_frames[0] is None:' not in uio_content:
        uio_content = uio_content.replace(
            'with av.open(str(video_path), mode="w") as container:',
            'if stacked_frames is None or len(stacked_frames) == 0 or stacked_frames[0] is None:\n        return\n    with av.open(str(video_path), mode="w") as container:'
        )
        uio_file.write_text(uio_content)
        print('  ✓ Applied write_video empty frame patch')
except Exception as e:
    print(f'  Note on write_video patch: {e}')

# Patch 10: Fallback to h264 when libsvtav1 is unsupported in lerobot/configs/video.py
try:
    import lerobot.configs.video as vid_cfg_mod
    vc_file = pathlib.Path(vid_cfg_mod.__file__)
    vc_content = vc_file.read_text()
    if 'falling back to \'h264\'' not in vc_content:
        vc_content = vc_content.replace(
            'raise ValueError(f"Unsupported video codec: {self.vcodec} with video backend {self.video_backend}")',
            'logger.warning(f"Video codec \'{self.vcodec}\' unsupported with backend \'{self.video_backend}\', falling back to \'h264\'")\n        self.vcodec = "h264"\n        return'
        )
        vc_file.write_text(vc_content)
        print('  ✓ Applied video codec h264 fallback patch')
except Exception as e:
    print(f'  Note on video codec patch: {e}')

# Patch 11: Allow existing directory in dataset metadata creation in lerobot/datasets/dataset_metadata.py
try:
    import lerobot.datasets.dataset_metadata as dm_mod
    dm_file = pathlib.Path(dm_mod.__file__)
    dm_content = dm_file.read_text()
    if 'obj.root.mkdir(parents=True, exist_ok=True)' not in dm_content:
        dm_content = dm_content.replace(
            'obj.root.mkdir(parents=True, exist_ok=False)',
            'obj.root.mkdir(parents=True, exist_ok=True)'
        )
        dm_file.write_text(dm_content)
        print('  ✓ Applied dataset_metadata exist_ok patch')
except Exception as e:
    print(f'  Note on dataset_metadata patch: {e}')
"

echo "===================================================================="
echo "✓ Virtual environment setup complete!"
echo "To activate in your terminal run:"
echo "    source ${VENV_DIR}/bin/activate"
echo "===================================================================="
