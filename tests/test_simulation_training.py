import json
from pathlib import Path

import pytest

from moira.robot_config import RobotModel
from moira.simulation_training import (
    create_simulation_training_template,
    simulation_training_preflight,
    validate_simulation_training_config,
)

MODEL = Path("robot_models/four_dof_desktop_arm/model.json")
PHYSICAL_MODEL = Path(
    "robot_models/four_dof_desktop_arm/physical_three_actuator_model.json"
)
MUJOCO = Path("robot_models/four_dof_desktop_arm/kinematic_validation.xml")
PHYSICAL_MUJOCO = Path(
    "robot_models/four_dof_desktop_arm/physical_three_actuator_validation.xml"
)


def _model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _complete_config():
    value = create_simulation_training_template(_model(), MUJOCO)
    value.update(timestep_s=0.01, rollout_horizon_s=2.5, parallel_candidates=4)
    for link in value["link_dynamics"].values():
        link.update(
            mass_kg=0.1,
            center_of_mass_m=[0, 0, 0],
            diaginertia_kg_m2=[0.001, 0.001, 0.001],
            friction=[0.8, 0.01, 0.001],
        )
    for joint in value["joint_dynamics"].values():
        joint.update(
            damping_nms_rad=0.01,
            armature_kg_m2=0,
            backlash_deg=1,
            max_torque_nm=0.2,
            time_constant_s=0.08,
        )
    value["domain_randomization"].update(
        mass_fraction=0.1,
        friction_fraction=0.1,
        actuator_delay_s=0.02,
        camera_translation_m=0.005,
        camera_rotation_deg=1,
    )
    return value


def test_simulation_template_tracks_exact_cad_assets_without_inventing_dynamics():
    value = create_simulation_training_template(_model(), MUJOCO)

    assert len(value["link_dynamics"]) == 7
    assert len(value["joint_dynamics"]) == 5
    assert all(item["mass_kg"] is None for item in value["link_dynamics"].values())
    with pytest.raises(TypeError, match="timestep_s"):
        validate_simulation_training_config(
            value, RobotModel.from_mapping(_model()), MUJOCO
        )


def test_complete_simulation_config_validates_and_stale_geometry_is_rejected():
    value = _complete_config()
    validated = validate_simulation_training_config(
        value, RobotModel.from_mapping(_model()), MUJOCO
    )
    assert validated["rollout_horizon_s"] == 2.5

    value["mujoco_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different MuJoCo model"):
        validate_simulation_training_config(
            value, RobotModel.from_mapping(_model()), MUJOCO
        )


def test_preflight_reports_incomplete_calibration_and_physics(tmp_path):
    simulation = tmp_path / "simulation.json"
    simulation.write_text(
        Path("config/simulation_training.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    camera = tmp_path / "incomplete-camera.json"
    camera_value = json.loads(
        Path("config/co6_camera_calibration.json").read_text(encoding="utf-8")
    )
    camera_value.update(
        captured_at=None,
        camera_matrix=None,
        distortion_coefficients=None,
        camera_to_base_matrix=None,
    )
    camera.write_text(json.dumps(camera_value), encoding="utf-8")
    report = simulation_training_preflight(
        PHYSICAL_MODEL,
        PHYSICAL_MUJOCO,
        Path("config/arm1_servo_calibration.json"),
        camera,
        simulation,
    )
    assert not report["ready"]
    assert "CAD-derived kinematics validation" not in report["issues"]
    assert "complete servo calibration" in report["issues"]
    assert "complete camera calibration" in report["issues"]
    assert "complete simulation dynamics configuration" in report["issues"]
    assert "physical payload validation" in report["issues"]
