#!/usr/bin/env python3
"""Simple test runner for teleoperation module.

Runs tests without pytest plugins to avoid ROS import issues.
"""

import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from teleoperation.devices import (
    DeviceState,
    MockDevice,
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
    TeleoperationApp,
)


def test_device_state():
    """Test DeviceState class."""
    state = DeviceState(
        axes={"x": 0.5, "y": -0.2},
        buttons={"button_0": True},
    )
    assert state.axes == {"x": 0.5, "y": -0.2}
    assert state.buttons == {"button_0": True}
    assert state.timestamp is not None

    state_dict = state.to_dict()
    assert "axes" in state_dict
    assert "buttons" in state_dict
    assert "timestamp" in state_dict
    print("✓ test_device_state passed")


def test_mock_device():
    """Test MockDevice."""
    device = MockDevice(name="test_mock", num_axes=4, num_buttons=2)
    assert device.name == "test_mock"
    assert device.initialize()
    assert device.is_connected

    device.set_axis_value(0, 0.5)
    device.set_button_state(0, True)

    state = device.get_state()
    assert state.axes["axis_0"] == 0.5
    assert state.buttons["button_0"] is True

    # Test clamping
    device.set_axis_value(1, 2.0)
    state = device.get_state()
    assert state.axes["axis_1"] == 1.0

    device.close()
    assert not device.is_connected
    print("✓ test_mock_device passed")


def test_device_manager():
    """Test DeviceManager."""
    manager = DeviceManager()

    device = manager.add_device("mock", name="test_mock")
    assert device is not None
    assert "test_mock" in manager.devices

    state = manager.get_device_state("test_mock")
    assert state is not None

    manager.add_device("mock", name="mock2")
    states = manager.get_all_states()
    assert len(states) == 2

    manager.close_all()
    assert len(manager.devices) == 0
    print("✓ test_device_manager passed")


def test_input_mapper():
    """Test InputToCommandMapper."""
    mapper = InputToCommandMapper(control_frame=ControlFrame.END_EFFECTOR)
    mapper.set_deadzone(0.1)
    mapper.set_mapping({"axis_0": "vel_x"})

    # Below deadzone
    command = mapper.map_input_to_command({"axis_0": 0.05}, {})
    assert command["vel_x"] == 0.0

    # Above deadzone
    command = mapper.map_input_to_command({"axis_0": 0.2}, {})
    assert command["vel_x"] == 0.2

    # Test gain
    mapper.set_gain(2.0)
    command = mapper.map_input_to_command({"axis_0": 0.5}, {})
    assert command["vel_x"] == 1.0

    # Test scale limits
    mapper.set_scale_limits({"vel_x": [-0.5, 0.5]})
    command = mapper.map_input_to_command({"axis_0": 1.0}, {})
    assert command["vel_x"] == 0.5
    print("✓ test_input_mapper passed")


def test_joint_velocity_mapper():
    """Test JointVelocityMapper."""
    mapper = JointVelocityMapper(num_joints=6)
    assert mapper.control_frame == ControlFrame.JOINT

    command = mapper.map_input_to_command(
        {"axis_0": 0.1, "axis_1": 0.2, "axis_2": 0.3},
        {}
    )
    assert command["joint_0"] == 0.1
    assert command["joint_1"] == 0.2
    assert command["joint_2"] == 0.3
    print("✓ test_joint_velocity_mapper passed")


def test_end_effector_mapper():
    """Test EndEffectorVelocityMapper."""
    mapper = EndEffectorVelocityMapper()
    assert mapper.control_frame == ControlFrame.END_EFFECTOR

    command = mapper.map_input_to_command(
        {"axis_0": 0.1, "axis_1": 0.2, "axis_2": 0.3},
        {}
    )
    assert command["vel_x"] == 0.1
    assert command["vel_y"] == 0.2
    assert command["vel_z"] == 0.3
    print("✓ test_end_effector_mapper passed")


def test_configurable_mapper():
    """Test ConfigurableMapper."""
    config = {
        "teleoperation": {
            "control_frame": "end_effector",
            "deadzone": 0.1,
            "gain": 2.0,
            "axis_mapping": {
                "axis_0": "vel_x",
                "axis_1": "vel_y",
            },
        }
    }

    mapper = ConfigurableMapper(config)
    assert mapper.mapper is not None

    command = mapper.map_input_to_command({"axis_0": 0.1}, {})
    assert "vel_x" in command
    print("✓ test_configurable_mapper passed")


def test_command_smoothing():
    """Test CommandSmoothing."""
    smoother = CommandSmoothing(smoothing_factor=0.0)

    command1 = {"vel_x": 1.0}
    result1 = smoother.smooth(command1)
    assert result1["vel_x"] == 1.0

    command2 = {"vel_x": 0.0}
    result2 = smoother.smooth(command2)
    assert result2["vel_x"] == 0.0

    # Test with smoothing
    smoother2 = CommandSmoothing(smoothing_factor=0.5)
    smoother2.smooth({"vel_x": 1.0})
    result = smoother2.smooth({"vel_x": 0.0})
    assert 0.4 < result["vel_x"] < 0.6

    smoother2.reset()
    result = smoother2.smooth({"vel_x": 0.0})
    assert result["vel_x"] == 0.0
    print("✓ test_command_smoothing passed")


def test_teleop_state():
    """Test TeleopState."""
    state = TeleopState()
    assert not state.is_running
    assert not state.is_recording
    assert not state.is_emergency_stopped
    assert state.recording_frames == []
    print("✓ test_teleop_state passed")


def test_teleop_app():
    """Test TeleoperationApp."""
    from teleoperation.teleop_app import TeleopEnvironmentInterface
    import numpy as np

    class MockEnv(TeleopEnvironmentInterface):
        def get_observation(self):
            return np.zeros(12)

        def step(self, action):
            return np.zeros(12), 0.0, False, {}

        def reset(self):
            return np.zeros(12)

        def close(self):
            pass

    env = MockEnv()
    app = TeleoperationApp(env)

    assert app.env is env
    assert app.polling_freq == 100.0
    assert app.render

    device = app.register_device("mock", name="test_mock")
    assert device is not None
    assert device.is_connected

    assert not app.state.is_recording
    app._start_recording()
    assert app.state.is_recording

    app._stop_recording()
    assert not app.state.is_recording

    assert not app.state.is_emergency_stopped
    app._emergency_stop()
    assert app.state.is_emergency_stopped
    print("✓ test_teleop_app passed")


def test_mapping_pipeline():
    """Test complete mapping pipeline."""
    device = MockDevice()
    device.initialize()
    device.set_axis_value(0, 0.5)
    device.set_axis_value(1, -0.3)

    state = device.get_state()

    mapper = EndEffectorVelocityMapper()
    mapper.set_deadzone(0.05)
    mapper.set_gain(1.0)

    command = mapper.map_input_to_command(state.axes, state.buttons)

    assert command["vel_x"] == 0.5
    assert command["vel_y"] == -0.3
    print("✓ test_mapping_pipeline passed")


def test_mapping_with_scaling():
    """Test mapping with scale limits."""
    device = MockDevice()
    device.initialize()
    device.set_axis_value(0, 1.0)

    state = device.get_state()

    mapper = EndEffectorVelocityMapper()
    mapper.set_gain(2.0)
    mapper.set_scale_limits({"vel_x": [-0.5, 0.5]})

    command = mapper.map_input_to_command(state.axes, state.buttons)

    # Should be clipped to 0.5
    assert command["vel_x"] == 0.5
    print("✓ test_mapping_with_scaling passed")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_device_state,
        test_mock_device,
        test_device_manager,
        test_input_mapper,
        test_joint_velocity_mapper,
        test_end_effector_mapper,
        test_configurable_mapper,
        test_command_smoothing,
        test_teleop_state,
        test_teleop_app,
        test_mapping_pipeline,
        test_mapping_with_scaling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed:")
            traceback.print_exc()
            print()

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
