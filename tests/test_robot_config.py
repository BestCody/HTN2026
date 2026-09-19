import json
from pathlib import Path

import pytest

from moira.components import ComponentRegistry
from moira.edge_components import LoadFeedbackComponent
from moira.physical import BimanualControlComponent, PhysicalAI
from moira.robot_config import RobotModel, load_bundled_robot_model, load_robot_model
from moira.robot_runtime import robot_bound_local_factories
from moira.specialists import (
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    PlanarBimanualIK,
    StateSpaceWorldModel,
    TactileSignalModel,
)
from tools.merge_fusion_robot_export import merge

MODEL_PATH = Path("robot_models/four_dof_desktop_arm/model.json")


def _motion_ready_mapping() -> dict[str, object]:
    value = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    value["payload_limit_kg"] = 0.025
    value["coordinate_frame"] = {"ground_plane": "XZ", "up_axis": "Y"}
    value["geometry"] = {
        "upper_arm_m": 0.16,
        "forearm_m": 0.15,
        "shoulder_height_m": 0.12,
    }
    value["meshes"] = {name: f"meshes/{name}.stl" for name in value["components"]}
    value["mesh_scale_to_m"] = 0.001
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
    axes = ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    for joint, axis in zip(value["joints"], axes, strict=True):
        joint.update(
            axis=axis,
            lower_deg=-45.0,
            upper_deg=45.0,
            home_deg=0.0,
            origin_m=[0.0, 0.0, 0.0],
            max_velocity_deg_s=90.0,
        )
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
            "right": {"physical_id": "arm_2", "installed": True},
        },
        "actuators": {
            arm: {
                joint["name"]: {
                    "channel": arm_index * 4 + joint_index,
                    "pulse_at_lower_us": 1000.0,
                    "pulse_at_upper_us": 2000.0,
                }
                for joint_index, joint in enumerate(value["joints"])
            }
            for arm_index, arm in enumerate(("left", "right"))
        },
    }
    for flag in (
        "calibration_complete",
        "kinematics_validated",
        "collision_geometry_validated",
        "actuator_mapping_validated",
        "payload_validated",
    ):
        value[flag] = True
    value["blockers"] = []
    return value


def test_bundled_robot_config_matches_canonical_robot_config():
    assert load_bundled_robot_model() == load_robot_model(MODEL_PATH)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 1, "schema_version"),
        ("servo_count", 3, "servo_count"),
        ("components", "not-a-list", "components"),
        ("blockers", "not-a-list", "blockers"),
    ),
)
def test_robot_config_rejects_malformed_structural_metadata(field, value, message):
    config = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    config[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        RobotModel.from_mapping(config)


def test_active_robot_profile_matches_new_sources_and_blocks_motion():
    model = load_robot_model(MODEL_PATH, verify_source=True)
    assert model.model_id == "four-dof-desktop-arm-v1"
    assert model.servo_models == {
        "J1_BASE_YAW": "MG996R",
        "J2_SHOULDER": "MG996R",
        "J3_ELBOW": "SG90",
        "J4_END_EFFECTOR": "SG90",
    }
    assert model.payload_limit_kg is None
    assert model.servo_controller.servo_supply_voltage == 6.0
    assert model.servo_controller.servo_supply_current_a == 10.0
    assert model.servo_controller.installed_arms == ()
    assert all(
        calibration.channel is None
        for mapping in model.servo_controller.actuators.values()
        for calibration in mapping.values()
    )
    assert [joint.name for joint in model.kinematic_joints] == [
        "J1_BASE_YAW",
        "J2_SHOULDER",
        "J3_ELBOW",
    ]
    assert model.gripper_joint.name == "J4_END_EFFECTOR"
    assert not model.motion_ready
    assert "validated payload limit" in model.readiness_issues
    assert "joint axes" in model.readiness_issues
    assert "joint limits and home positions" in model.readiness_issues
    with pytest.raises(RuntimeError, match="not motion-ready"):
        model.require_motion_ready()


def test_motion_components_use_calibrated_robot_profile():
    model = RobotModel.from_mapping(_motion_ready_mapping())
    assert model.motion_ready

    ik = PlanarBimanualIK.from_robot_model(model)
    trajectory = BoundedTrajectoryPlanner.from_robot_model(model)
    collision = ConservativeCollisionChecker.from_robot_model(model)
    world_model = StateSpaceWorldModel.from_robot_model(model, "forward-dynamics")
    safety = HardSafetyRiskModel.from_robot_model(model)
    feedback = LoadFeedbackComponent.from_robot_model(model)
    tactile = TactileSignalModel.from_robot_model(model)

    class Driver:
        def execute(self, step, world):
            raise AssertionError("not executed")

        def stop(self):
            return None

    controller = BimanualControlComponent.from_robot_model(Driver(), Driver(), model)
    factories = robot_bound_local_factories(model, Driver(), Driver())
    system = PhysicalAI.from_robot_model(ComponentRegistry(ram_budget_mb=64), model)
    assert not ik.include_wrist_joint
    assert ik.up_axis == "y"
    assert len(ik.joint_limits_rad) == 3
    assert len(trajectory.joint_limits_rad) == 3
    assert trajectory.frequency_hz == 20.0
    assert trajectory.max_gripper_velocity_m_s == 0.03
    assert trajectory.max_gripper_force_n == 3.0
    assert len(collision.joint_limits_rad) == 3
    assert collision.required_clearance_m == 0.02
    assert world_model.per_arm_payload_kg == model.payload_limit_kg
    assert safety.per_arm_payload_kg == model.payload_limit_kg
    assert feedback.single_arm_payload_kg == model.payload_limit_kg
    assert tactile.min_stability_score == 0.70
    assert controller.timeout_margin_seconds == 0.5
    assert set(factories) == {
        "local-bimanual-ik",
        "local-trajectory-planner",
        "local-collision-checker",
        "local-safety-risk",
        "dual-arm-hardware",
        "local-failure-classifier",
        "local-load-feedback",
    }
    assert system.robot_model_id == model.model_id
    assert system.gripper_geometry["max_width_m"] == 0.08


def test_fusion_export_merge_populates_geometry_origins_axes_and_meshes(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    export_dir = tmp_path / "fusion_export"
    mesh_dir = export_dir / "meshes"
    mesh_dir.mkdir(parents=True)
    base = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    meshes = {}
    for component in base["components"]:
        path = mesh_dir / f"{component}.stl"
        path.write_bytes(b"solid test\nendsolid test\n")
        meshes[component] = f"meshes/{path.name}"
    exported = {
        "complete": True,
        "mesh_units": "unitless",
        "mesh_scale_to_m": None,
        "meshes": meshes,
        "parameters": [
            {"name": "upper_length", "unit": "mm", "database_value": 16.0},
            {"name": "forearm_length", "unit": "mm", "database_value": 15.0},
            {"name": "shoulder_height", "unit": "mm", "database_value": 12.0},
        ],
        "joints": [
            {
                "name": joint["name"],
                "axis": [0.0, 0.0, 1.0],
                "transform": {"translation_m": [0.01 * index, 0.0, 0.0]},
                "limits": {
                    "minimum_enabled": True,
                    "minimum_rad": -0.5,
                    "maximum_enabled": True,
                    "maximum_rad": 0.5,
                    "rest_enabled": True,
                    "rest_rad": 0.0,
                },
            }
            for index, joint in enumerate(base["joints"])
        ],
    }
    export_path = export_dir / "fusion_robot_export.json"
    export_path.write_text(json.dumps(exported), encoding="utf-8")
    output = tmp_path / "generated" / "model.json"
    merge(model_path, export_path, output)

    model = load_robot_model(output)
    assert model.upper_arm_m == 0.16
    assert model.forearm_m == 0.15
    assert model.shoulder_height_m == 0.12
    assert all(joint.origin_m is not None for joint in model.joints)
    assert all(joint.axis is not None for joint in model.joints)
    assert len(model.meshes) == 5
    assert all((output.parent / path).is_file() for path in model.meshes.values())
    assert model.mesh_scale_to_m is None
    assert any("STL scale" in blocker for blocker in model.blockers)
