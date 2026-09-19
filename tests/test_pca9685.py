import json
from pathlib import Path

import pytest

from moira.pca9685 import (
    PCA9685ArmDriver,
    pca9685_arm_drivers,
    pca9685_installed_arm_drivers,
)
from moira.physical import ActionChunk, MotionTrajectory, TrajectoryPoint, WorldState
from moira.robot_config import RobotModel

MODEL_PATH = Path("robot_models/current_lightweight_arm/model.json")


def _calibrated_model(*, right_installed: bool = True) -> RobotModel:
    value = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    value["geometry"] = {
        "upper_arm_m": 0.16,
        "forearm_m": 0.15,
        "shoulder_height_m": 0.12,
    }
    value["bimanual_mount"] = {"shoulder_offset_m": 0.20}
    value["control_limits"] = {
        "trajectory_frequency_hz": 20.0,
        "required_clearance_m": 0.02,
        "max_gripper_width_m": 0.08,
        "max_gripper_velocity_m_s": 0.03,
        "max_gripper_force_n": 3.0,
        "controller_timeout_margin_s": 0.5,
    }
    value["safety_limits"] = {
        "max_slip_probability": 0.30,
        "max_collision_probability": 0.15,
        "min_grasp_stability_score": 0.70,
    }
    value["meshes"] = {name: f"meshes/{name}.stl" for name in value["components"]}
    value["mesh_scale_to_m"] = 0.001
    for flag in (
        "calibration_complete",
        "kinematics_validated",
        "collision_geometry_validated",
        "actuator_mapping_validated",
        "payload_validated",
    ):
        value[flag] = True
    value["blockers"] = []
    for joint in value["joints"]:
        joint["origin_m"] = [0.0, 0.0, 0.0]
        joint["max_velocity_deg_s"] = 90.0
    value["servo_controller"] = {
        "type": "pca9685",
        "i2c_address": 0x40,
        "reference_clock_hz": 25_000_000.0,
        "pwm_frequency_hz": 50.0,
        "servo_supply_voltage": 6.0,
        "servo_supply_current_a": 10.0,
        "servo_power_validated": True,
        "output_enable_validated": True,
        "arm_installations": {
            "left": {"physical_id": "arm_1", "installed": True},
            "right": {"physical_id": "arm_2", "installed": right_installed},
        },
        "actuators": {
            arm: {
                joint["name"]: {
                    "channel": (
                        arm_index * 4 + joint_index
                        if arm == "left" or right_installed
                        else None
                    ),
                    "pulse_at_lower_us": 1000.0 if arm == "left" or right_installed else None,
                    "pulse_at_upper_us": 2000.0 if arm == "left" or right_installed else None,
                }
                for joint_index, joint in enumerate(value["joints"])
            }
            for arm_index, arm in enumerate(("left", "right"))
        },
    }
    return RobotModel.from_mapping(value)


class FakePCA9685:
    frequency_hz = 50.0

    def __init__(self):
        self.commands = []
        self.disabled = []

    def set_pulses_us(self, pulses):
        self.commands.append(dict(pulses))

    def disable_channels(self, channels):
        self.disabled.append(tuple(channels))


def test_pca9685_driver_maps_robot_limits_to_configured_pulse_endpoints():
    model = _calibrated_model()
    device = FakePCA9685()
    driver = PCA9685ArmDriver("left", model, device)
    joints = tuple(joint.upper_rad for joint in model.kinematic_joints)
    trajectory = MotionTrajectory(
        "plan",
        (
            TrajectoryPoint(
                0.001,
                {"left": joints},
                "chunk",
                {"left": model.max_gripper_width_m},
            ),
        ),
        0.001,
    )
    chunk = ActionChunk(
        "chunk",
        "step",
        "grasp",
        ("left",),
        0.001,
        None,
        (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
        model.max_gripper_width_m,
        1.0,
    )
    telemetry = driver.execute_chunk(chunk, trajectory, WorldState((), {}))
    assert telemetry.success
    assert device.commands == [{0: 2000.0, 1: 2000.0, 2: 2000.0, 3: 2000.0}]

    driver.stop()
    assert device.disabled == [(0, 1, 2, 3)]
    with pytest.raises(RuntimeError, match="latched"):
        driver.execute_chunk(chunk, trajectory, WorldState((), {}))
    driver.rearm()
    assert device.disabled[-1] == (0, 1, 2, 3)


def test_pca9685_pair_shares_device_and_rejects_wrong_frequency():
    model = _calibrated_model()
    device = FakePCA9685()
    left, right, shared = pca9685_arm_drivers(model, device)
    assert left.device is right.device is shared

    device.frequency_hz = 60.0
    with pytest.raises(ValueError, match="frequency"):
        PCA9685ArmDriver("left", model, device)


def test_pca9685_configuration_rejects_duplicate_channels():
    value = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    value["servo_controller"]["actuators"]["left"]["J1_BASE_YAW"]["channel"] = 0
    value["servo_controller"]["actuators"]["right"]["J1_BASE_YAW"]["channel"] = 0
    with pytest.raises(ValueError, match="unique"):
        RobotModel.from_mapping(value)


def test_pca9685_pair_rejects_missing_second_physical_arm():
    model = _calibrated_model(right_installed=False)
    assert model.motion_ready
    assert not model.bimanual_motion_ready
    assert model.servo_controller.installed_arms == ("left",)
    assert model.bimanual_readiness_issues == ("arm_2 is not built",)
    with pytest.raises(RuntimeError, match="not bimanual-motion-ready"):
        pca9685_arm_drivers(model, FakePCA9685())

    drivers, shared = pca9685_installed_arm_drivers(model, FakePCA9685())
    assert tuple(drivers) == ("left",)
    assert drivers["left"].physical_arm_id == "arm_1"
    assert drivers["left"].device is shared
