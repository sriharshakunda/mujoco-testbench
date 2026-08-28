"""Unit tests for teleoperation module.

Tests device abstraction, input mapping, and teleoperation application.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock, patch

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from teleoperation.devices import (
    DeviceState,
    MockDevice,
    SpacemouseDevice,
    JoystickDevice,
    DeviceManager,
)
from teleoperation.mapping import (
    ControlFrame,
    InputToCommandMapper,
    JointVelocityMapper,
    EndEffectorVelocityMapper,
    ConfigurableMapper,
    CommandSmoothing,
)
from teleoperation.teleop_app import (
    TeleopState,
    TeleopEnvironmentInterface,
    TeleoperationApp,
    create_teleop_app_from_config,
)


class TestDeviceState:
    """Tests for DeviceState class."""

    def test_device_state_creation(self):
        """Test creating device state."""
        state = DeviceState(
            axes={"x": 0.5, "y": -0.2},
            buttons={"button_0": True},
        )
        assert state.axes == {"x": 0.5, "y": -0.2}
        assert state.buttons == {"button_0": True}
        assert state.timestamp is not None

    def test_device_state_defaults(self):
        """Test device state with default values."""
        state = DeviceState()
        assert state.axes == {}
        assert state.buttons == {}
        assert state.timestamp is not None

    def test_device_state_to_dict(self):
        """Test converting device state to dict."""
        state = DeviceState(
            axes={"x": 0.5},
            buttons={"button_0": True},
        )
        state_dict = state.to_dict()
        assert "axes" in state_dict
        assert "buttons" in state_dict
        assert "timestamp" in state_dict


class TestMockDevice:
    """Tests for MockDevice."""

    def test_mock_device_initialization(self):
        """Test initializing mock device."""
        device = MockDevice(name="test_mock", num_axes=4, num_buttons=2)
        assert device.name == "test_mock"
        assert device.initialize()
        assert device.is_connected

    def test_mock_device_get_state(self):
        """Test getting state from mock device."""
        device = MockDevice(num_axes=3, num_buttons=2)
        device.initialize()

        state = device.get_state()
        assert isinstance(state, DeviceState)
        assert len(state.axes) == 3
        assert len(state.buttons) == 2

    def test_mock_device_set_values(self):
        """Test setting mock axis and button values."""
        device = MockDevice(num_axes=2, num_buttons=2)
        device.initialize()

        device.set_axis_value(0, 0.5)
        device.set_button_state(0, True)

        state = device.get_state()
        assert state.axes["axis_0"] == 0.5
        assert state.buttons["button_0"] is True

    def test_mock_device_clamp_values(self):
        """Test that mock device clamps values to [-1, 1]."""
        device = MockDevice()
        device.initialize()

        device.set_axis_value(0, 2.0)  # Should clamp to 1.0
        device.set_axis_value(1, -2.0)  # Should clamp to -1.0

        state = device.get_state()
        assert state.axes["axis_0"] == 1.0
        assert state.axes["axis_1"] == -1.0

    def test_mock_device_close(self):
        """Test closing mock device."""
        device = MockDevice()
        device.initialize()
        assert device.is_connected

        device.close()
        assert not device.is_connected


class TestDeviceManager:
    """Tests for DeviceManager."""

    def test_device_manager_add_device(self):
        """Test adding device to manager."""
        manager = DeviceManager()
        device = manager.add_device("mock", name="test_mock")

        assert device is not None
        assert "test_mock" in manager.devices
        assert device.is_connected

    def test_device_manager_get_device_state(self):
        """Test getting state from managed device."""
        manager = DeviceManager()
        manager.add_device("mock", name="test_mock")

        state = manager.get_device_state("test_mock")
        assert state is not None
        assert isinstance(state, DeviceState)

    def test_device_manager_get_all_states(self):
        """Test getting states from all devices."""
        manager = DeviceManager()
        manager.add_device("mock", name="mock1")
        manager.add_device("mock", name="mock2")

        states = manager.get_all_states()
        assert len(states) == 2
        assert "mock1" in states
        assert "mock2" in states

    def test_device_manager_close_all(self):
        """Test closing all managed devices."""
        manager = DeviceManager()
        manager.add_device("mock", name="mock1")
        manager.add_device("mock", name="mock2")

        manager.close_all()
        assert len(manager.devices) == 0


class TestInputToCommandMapper:
    """Tests for InputToCommandMapper."""

    def test_mapper_creation(self):
        """Test creating input mapper."""
        mapper = InputToCommandMapper(control_frame=ControlFrame.END_EFFECTOR)
        assert mapper.control_frame == ControlFrame.END_EFFECTOR

    def test_mapper_deadzone(self):
        """Test deadzone filtering."""
        mapper = InputToCommandMapper()
        mapper.set_deadzone(0.1)
        mapper.set_mapping({"axis_0": "vel_x"})

        # Below deadzone
        command = mapper.map_input_to_command(
            {"axis_0": 0.05},
            {}
        )
        assert command["vel_x"] == 0.0

        # Above deadzone
        command = mapper.map_input_to_command(
            {"axis_0": 0.2},
            {}
        )
        assert command["vel_x"] == 0.2

    def test_mapper_gain(self):
        """Test gain scaling."""
        mapper = InputToCommandMapper()
        mapper.set_gain(2.0)
        mapper.set_mapping({"axis_0": "vel_x"})

        command = mapper.map_input_to_command({"axis_0": 0.5}, {})
        assert command["vel_x"] == 1.0  # 0.5 * 2.0

    def test_mapper_scale_limits(self):
        """Test scaling limit enforcement."""
        mapper = InputToCommandMapper()
        mapper.set_gain(2.0)
        mapper.set_scale_limits({"vel_x": [-0.5, 0.5]})
        mapper.set_mapping({"axis_0": "vel_x"})

        # Input 1.0 * gain 2.0 = 2.0, should clamp to 0.5
        command = mapper.map_input_to_command({"axis_0": 1.0}, {})
        assert command["vel_x"] == 0.5

    def test_mapper_multiple_axes(self):
        """Test mapping multiple axes."""
        mapper = InputToCommandMapper()
        mapper.set_mapping({
            "axis_0": "vel_x",
            "axis_1": "vel_y",
            "axis_2": "vel_z",
        })

        command = mapper.map_input_to_command(
            {"axis_0": 0.1, "axis_1": 0.2, "axis_2": 0.3},
            {}
        )
        assert command["vel_x"] == 0.1
        assert command["vel_y"] == 0.2
        assert command["vel_z"] == 0.3

    def test_mapper_unknown_axis(self):
        """Test handling unknown axes."""
        mapper = InputToCommandMapper()
        mapper.set_mapping({"axis_0": "vel_x"})

        # Unknown axis should be ignored
        command = mapper.map_input_to_command(
            {"axis_0": 0.1, "axis_99": 0.5},
            {}
        )
        assert "vel_x" in command
        assert "axis_99" not in command


class TestJointVelocityMapper:
    """Tests for JointVelocityMapper."""

    def test_joint_velocity_mapper_default_mapping(self):
        """Test default joint velocity mapping."""
        mapper = JointVelocityMapper(num_joints=6)
        assert mapper.control_frame == ControlFrame.JOINT

        command = mapper.map_input_to_command(
            {"axis_0": 0.1, "axis_1": 0.2, "axis_2": 0.3},
            {}
        )
        assert command["joint_0"] == 0.1
        assert command["joint_1"] == 0.2
        assert command["joint_2"] == 0.3


class TestEndEffectorVelocityMapper:
    """Tests for EndEffectorVelocityMapper."""

    def test_end_effector_mapper_default_mapping(self):
        """Test default end-effector velocity mapping."""
        mapper = EndEffectorVelocityMapper()
        assert mapper.control_frame == ControlFrame.END_EFFECTOR

        command = mapper.map_input_to_command(
            {"axis_0": 0.1, "axis_1": 0.2, "axis_2": 0.3},
            {}
        )
        assert command["vel_x"] == 0.1
        assert command["vel_y"] == 0.2
        assert command["vel_z"] == 0.3


class TestConfigurableMapper:
    """Tests for ConfigurableMapper."""

    def test_configurable_mapper_from_config(self):
        """Test creating mapper from configuration."""
        config = {
            "teleoperation": {
                "control_frame": "end_effector",
                "deadzone": 0.1,
                "gain": 2.0,
                "axis_mapping": {
                    "axis_0": "vel_x",
                    "axis_1": "vel_y",
                },
                "scale_limits": {
                    "vel_x": [-0.5, 0.5],
                    "vel_y": [-0.5, 0.5],
                },
            }
        }

        mapper = ConfigurableMapper(config)
        assert mapper.mapper is not None

        command = mapper.map_input_to_command({"axis_0": 0.1}, {})
        assert "vel_x" in command

    def test_configurable_mapper_joint_control(self):
        """Test configurable mapper with joint control."""
        config = {
            "teleoperation": {
                "control_frame": "joint",
                "num_joints": 3,
            }
        }

        mapper = ConfigurableMapper(config)
        assert mapper.mapper.control_frame == ControlFrame.JOINT


class TestCommandSmoothing:
    """Tests for CommandSmoothing."""

    def test_command_smoothing_no_smoothing(self):
        """Test command smoothing with no smoothing."""
        smoother = CommandSmoothing(smoothing_factor=0.0)

        command1 = {"vel_x": 1.0}
        result1 = smoother.smooth(command1)
        assert result1["vel_x"] == 1.0

        command2 = {"vel_x": 0.0}
        result2 = smoother.smooth(command2)
        assert result2["vel_x"] == 0.0

    def test_command_smoothing_full_smoothing(self):
        """Test command smoothing with full smoothing."""
        smoother = CommandSmoothing(smoothing_factor=1.0)

        command1 = {"vel_x": 1.0}
        result1 = smoother.smooth(command1)
        assert result1["vel_x"] == 1.0

        command2 = {"vel_x": 0.0}
        result2 = smoother.smooth(command2)
        # With full smoothing, output should not change
        assert result2["vel_x"] == 1.0

    def test_command_smoothing_partial(self):
        """Test command smoothing with partial smoothing."""
        smoother = CommandSmoothing(smoothing_factor=0.5)

        command1 = {"vel_x": 1.0}
        smoother.smooth(command1)

        command2 = {"vel_x": 0.0}
        result2 = smoother.smooth(command2)
        # Should be average of 1.0 and 0.0
        assert 0.4 < result2["vel_x"] < 0.6

    def test_command_smoothing_reset(self):
        """Test resetting smoothing state."""
        smoother = CommandSmoothing(smoothing_factor=1.0)

        smoother.smooth({"vel_x": 1.0})
        smoother.reset()

        result = smoother.smooth({"vel_x": 0.0})
        assert result["vel_x"] == 0.0


class TestTeleopState:
    """Tests for TeleopState."""

    def test_teleop_state_creation(self):
        """Test creating teleoperation state."""
        state = TeleopState()
        assert not state.is_running
        assert not state.is_recording
        assert not state.is_emergency_stopped
        assert state.recording_frames == []


class MockEnvironment(TeleopEnvironmentInterface):
    """Mock environment for testing."""

    def __init__(self):
        """Initialize mock environment."""
        self.last_action = None
        self.step_count = 0

    def get_observation(self) -> np.ndarray:
        """Get observation from environment."""
        return np.zeros(12)  # 6 joints * 2 (pos, vel)

    def step(self, action: dict) -> tuple:
        """Execute one step."""
        self.last_action = action
        self.step_count += 1
        obs = self.get_observation()
        return obs, 0.0, False, {}

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.step_count = 0
        return self.get_observation()

    def render(self) -> None:
        """Render environment."""
        pass

    def close(self) -> None:
        """Close environment."""
        pass


class TestTeleoperationApp:
    """Tests for TeleoperationApp."""

    def test_teleop_app_creation(self):
        """Test creating teleoperation app."""
        env = MockEnvironment()
        app = TeleoperationApp(env)

        assert app.env is env
        assert app.polling_freq == 100.0
        assert app.render

    def test_teleop_app_register_device(self):
        """Test registering device with app."""
        env = MockEnvironment()
        app = TeleoperationApp(env)

        device = app.register_device("mock", name="test_mock")
        assert device is not None
        assert device.is_connected

    def test_teleop_app_recording(self):
        """Test trajectory recording."""
        env = MockEnvironment()
        app = TeleoperationApp(env)
        app.register_device("mock")

        assert not app.state.is_recording
        app._start_recording()
        assert app.state.is_recording

        app._stop_recording()
        assert not app.state.is_recording

    def test_teleop_app_emergency_stop(self):
        """Test emergency stop."""
        env = MockEnvironment()
        app = TeleoperationApp(env)
        app.register_device("mock")

        assert not app.state.is_emergency_stopped
        app._emergency_stop()
        assert app.state.is_emergency_stopped

    def test_teleop_app_callbacks(self):
        """Test callback registration and execution."""
        env = MockEnvironment()
        app = TeleoperationApp(env)

        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        app.register_on_record_start(test_callback)
        app._start_recording()
        assert callback_called

    def test_teleop_app_save_recording(self):
        """Test saving recording to file."""
        env = MockEnvironment()
        app = TeleoperationApp(env)
        app.register_device("mock")

        # Add some fake recorded frames
        app.state.is_recording = True
        for i in range(5):
            frame = {
                "timestamp": float(i),
                "observation": np.zeros(12),
                "action": {"vel_x": 0.1},
            }
            app.state.recording_frames.append(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recording.npz"
            app.save_recording(output_path)
            assert output_path.exists()

    def test_create_teleop_app_from_config(self):
        """Test creating app from config."""
        config = {
            "teleoperation": {
                "polling_freq": 50,
                "render": False,
                "devices": [{"type": "mock"}],
                "control_frame": "joint",
            }
        }

        env = MockEnvironment()
        app = create_teleop_app_from_config(env, config)

        assert app.polling_freq == 50
        assert not app.render
        assert len(app.device_manager.devices) > 0


class TestDeviceIntegration:
    """Integration tests for device handling."""

    def test_spacemouse_device_fallback(self):
        """Test spacemouse device with fallback to mock."""
        manager = DeviceManager()

        # Try spacemouse (likely to fail without hardware)
        spacemouse = manager.add_device("spacemouse")
        if spacemouse is None or not spacemouse.is_connected:
            # Add mock as fallback
            mock = manager.add_device("mock")
            assert mock is not None
            assert mock.is_connected

    def test_device_manager_multi_device(self):
        """Test managing multiple devices."""
        manager = DeviceManager()

        # Add multiple mock devices
        for i in range(3):
            device = manager.add_device("mock", name=f"mock_{i}")
            assert device is not None

        assert len(manager.devices) == 3

        # Get states from all
        states = manager.get_all_states()
        assert len(states) == 3

        # Close all
        manager.close_all()
        assert len(manager.devices) == 0


class TestMappingIntegration:
    """Integration tests for input mapping."""

    def test_mapping_pipeline(self):
        """Test complete mapping pipeline."""
        # Create device with values
        device = MockDevice()
        device.initialize()
        device.set_axis_value(0, 0.5)
        device.set_axis_value(1, -0.3)

        # Get device state
        state = device.get_state()

        # Map to command
        mapper = EndEffectorVelocityMapper()
        mapper.set_deadzone(0.05)
        mapper.set_gain(1.0)

        command = mapper.map_input_to_command(state.axes, state.buttons)

        assert command["vel_x"] == 0.5
        assert command["vel_y"] == -0.3

    def test_mapping_with_scaling(self):
        """Test mapping with scale limits."""
        device = MockDevice()
        device.initialize()
        device.set_axis_value(0, 1.0)

        state = device.get_state()

        mapper = EndEffectorVelocityMapper()
        mapper.set_gain(2.0)  # Will produce 2.0 output
        mapper.set_scale_limits({"vel_x": [-0.5, 0.5]})

        command = mapper.map_input_to_command(state.axes, state.buttons)

        # Should be clipped to 0.5
        assert command["vel_x"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
