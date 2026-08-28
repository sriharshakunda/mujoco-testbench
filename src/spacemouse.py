"""
SpaceMouse (3Dconnexion) 6-DOF Input Controller for Linux.
---------------------------------------------------------
Reads 6-DOF motion (Translation: X, Y, Z | Rotation: Roll, Pitch, Yaw)
and hardware buttons from 3Dconnexion SpaceNavigator/SpaceMouse devices
via the Linux evdev /dev/input subsystem with zero third-party dependencies.
"""

import os
import glob
import struct
import threading
import time
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Linux input event struct format: (timeval sec, timeval usec, type, code, value)
EVENT_FORMAT = "qqHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

# Linux Event Types
EV_SYN = 0x00
EV_KEY = 0x01
EV_REL = 0x02
EV_ABS = 0x03

# Axis codes (SpaceMouse sends EV_REL or EV_ABS)
REL_X = 0x00
REL_Y = 0x01
REL_Z = 0x02
REL_RX = 0x03
REL_RY = 0x04
REL_RZ = 0x05

# Button codes
BTN_0 = 0x100
BTN_1 = 0x101


class SpaceMouse:
    """
    Asynchronous 6-DOF SpaceMouse driver with clean linear responsiveness.
    """

    def __init__(
        self,
        deadzone_trans: float = 0.05,
        deadzone_rot: float = 0.07,
        translation_scale: float = 2.0 / 3500.0,
        rotation_scale: float = 1.5 / 3500.0,
    ):
        self.deadzone_trans = deadzone_trans
        self.deadzone_rot = deadzone_rot
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._dev_path: Optional[str] = None
        self.device_name = "None"

        self._lock = threading.Lock()
        # [x, y, z, roll, pitch, yaw] normalized to [-1.0, 1.0]
        self._axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # [btn_left, btn_right]
        self._buttons = [0, 0]
        self._last_event_time = time.time()

        self._find_and_open_device()

    def _find_and_open_device(self) -> bool:
        """Scan system and open 3Dconnexion device node in /dev/input."""
        candidates = []

        # 1. Search /dev/input/by-id
        candidates.extend(glob.glob("/dev/input/by-id/*Space*"))
        candidates.extend(glob.glob("/dev/input/by-id/*3Dconnexion*"))
        candidates.extend(glob.glob("/dev/input/by-id/*SpaceNavigator*"))

        # 2. Search /sys/class/input/event* device names
        for p in sorted(glob.glob("/dev/input/event*")):
            try:
                num = p.replace("/dev/input/event", "")
                sys_name_path = f"/sys/class/input/event{num}/device/name"
                if os.path.exists(sys_name_path):
                    with open(sys_name_path, "r") as f:
                        name = f.read().strip()
                    if "Space" in name or "3Dconnexion" in name:
                        if p not in candidates:
                            candidates.append(p)
            except Exception:
                pass

        for dev_path in candidates:
            try:
                fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
                self._fd = fd
                self._dev_path = dev_path
                self.device_name = os.path.basename(dev_path)
                logger.info("Connected to SpaceMouse at %s", dev_path)
                return True
            except PermissionError:
                if self._dev_path != dev_path:
                    self._dev_path = dev_path
                    print(
                        f"\n[SpaceMouse] Detected {dev_path} but permission was denied.\n"
                        f"To enable SpaceMouse, run: sudo chmod a+rw /dev/input/event* /dev/input/by-id/*\n"
                    )
            except Exception as e:
                logger.debug("Failed opening candidate %s: %s", dev_path, e)

        return False

    @property
    def is_connected(self) -> bool:
        return self._fd is not None

    def start(self) -> None:
        """Start reading input events in a background thread."""
        if not self.is_connected:
            if not self._find_and_open_device():
                return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        raw_axes = {
            REL_X: 0, REL_Y: 0, REL_Z: 0,
            REL_RX: 0, REL_RY: 0, REL_RZ: 0,
        }

        while self._running and self._fd is not None:
            try:
                data = os.read(self._fd, EVENT_SIZE * 16)
                if not data:
                    time.sleep(0.005)
                    continue

                for i in range(0, len(data), EVENT_SIZE):
                    chunk = data[i : i + EVENT_SIZE]
                    if len(chunk) < EVENT_SIZE:
                        continue
                    sec, usec, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, chunk)

                    if ev_type in (EV_REL, EV_ABS):
                        if ev_code in raw_axes:
                            raw_axes[ev_code] = ev_value
                    elif ev_type == EV_KEY:
                        with self._lock:
                            if ev_code == BTN_0:
                                self._buttons[0] = ev_value
                            elif ev_code == BTN_1:
                                self._buttons[1] = ev_value

                    elif ev_type == EV_SYN:
                        def apply_dz(val: float, scale: float, deadzone: float) -> float:
                            scaled = val * scale
                            if abs(scaled) <= deadzone:
                                return 0.0
                            norm = (abs(scaled) - deadzone) / (1.0 - deadzone + 1e-6)
                            return float(np.copysign(np.clip(norm, 0.0, 1.0), scaled))

                        with self._lock:
                            # Translation
                            self._axes[0] = apply_dz(raw_axes[REL_X], self.translation_scale, self.deadzone_trans)
                            self._axes[1] = apply_dz(-raw_axes[REL_Y], self.translation_scale, self.deadzone_trans)
                            self._axes[2] = apply_dz(-raw_axes[REL_Z], self.translation_scale, self.deadzone_trans)

                            # Rotation
                            self._axes[3] = apply_dz(raw_axes[REL_RX], self.rotation_scale, self.deadzone_rot)
                            self._axes[4] = apply_dz(raw_axes[REL_RY], self.rotation_scale, self.deadzone_rot)
                            self._axes[5] = apply_dz(-raw_axes[REL_RZ], self.rotation_scale, self.deadzone_rot)
                            self._last_event_time = time.time()

            except (BlockingIOError, InterruptedError):
                time.sleep(0.002)
            except Exception as e:
                logger.error("Error reading SpaceMouse: %s", e)
                time.sleep(0.01)

    def get_motion_state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return current normalized (x, y, z, pitch, yaw, roll) in range [-1.0, 1.0].
        Auto-decays to zero if no events are received within 150ms.
        """
        with self._lock:
            if time.time() - self._last_event_time > 0.15:
                self._axes = [0.0] * 6
            return tuple(self._axes)  # type: ignore

    def get_axes(self) -> Tuple[float, float, float, float, float, float]:
        """Alias for get_motion_state()."""
        return self.get_motion_state()

    def get_buttons(self) -> Tuple[int, int]:
        """Return (btn_left, btn_right) states (0 = released, 1 = pressed)."""
        with self._lock:
            return tuple(self._buttons)  # type: ignore

    def stop(self) -> None:
        self._running = False
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
