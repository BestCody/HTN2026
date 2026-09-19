"""Compile and exercise a generated MoIRA MuJoCo kinematic model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


def validate(model_path: Path, *, steps: int = 100) -> dict[str, object]:
    import mujoco

    coupling = ET.parse(model_path).getroot().find("equality/joint")
    if coupling is None:
        raise ValueError("MuJoCo model is missing the gripper gear coupling")
    coefficients = tuple(float(value) for value in coupling.attrib["polycoef"].split())
    if (
        len(coefficients) != 5
        or coefficients[1] == 0
        or any(coefficients[index] for index in range(2, 5))
    ):
        raise ValueError("Gripper gear coupling must be a nonzero linear relationship")
    coupling_ratio = coefficients[1]
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    expected = {
        "nq": 5,
        "nv": 5,
        "nbody": 7,
        "ngeom": 7,
        "nmesh": 7,
        "nu": 0,
        "neq": 1,
    }
    actual = {name: int(getattr(model, name)) for name in expected}
    if actual != expected:
        raise ValueError(f"Unexpected MuJoCo topology: {actual}, expected {expected}")

    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "turntable",
            "upper_arm",
            "forearm",
            "gripper_primary",
            "gripper_mirror",
        )
    }
    if any(value < 0 for value in body_ids.values()):
        raise ValueError("MuJoCo model is missing a canonical robot body")

    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    home_positions = {name: data.xpos[index].copy() for name, index in body_ids.items()}
    home_rotations = {name: data.xmat[index].copy() for name, index in body_ids.items()}
    motion_checks = {}
    checks = (
        ("J1_BASE_YAW", "upper_arm", "turntable", False),
        ("J2_SHOULDER", "forearm", "upper_arm", False),
        ("J3_ELBOW", "gripper_primary", "forearm", False),
        ("J4_END_EFFECTOR", "gripper_primary", "forearm", True),
    )
    qpos_addresses = {}
    for name in (
        "J1_BASE_YAW",
        "J2_SHOULDER",
        "J3_ELBOW",
        "J4_END_EFFECTOR",
        "J4_END_EFFECTOR_MIRROR",
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}")
        qpos_addresses[name] = int(model.jnt_qposadr[joint_id])
    for joint_name, moving_body, fixed_body, rotation_only in checks:
        data.qpos[:] = 0
        data.qpos[qpos_addresses[joint_name]] = 0.25
        if joint_name == "J4_END_EFFECTOR":
            data.qpos[qpos_addresses["J4_END_EFFECTOR_MIRROR"]] = 0.25 / coupling_ratio
        mujoco.mj_forward(model, data)
        fixed_delta = math.sqrt(
            sum(
                float(data.xpos[body_ids[fixed_body]][axis] - home_positions[fixed_body][axis]) ** 2
                for axis in range(3)
            )
        )
        if rotation_only:
            moving_delta = math.sqrt(
                sum(
                    float(
                        data.xmat[body_ids[moving_body]][index] - home_rotations[moving_body][index]
                    )
                    ** 2
                    for index in range(9)
                )
            )
        else:
            moving_delta = math.sqrt(
                sum(
                    float(
                        data.xpos[body_ids[moving_body]][axis] - home_positions[moving_body][axis]
                    )
                    ** 2
                    for axis in range(3)
                )
            )
        if fixed_delta > 1e-9 or moving_delta < 1e-5:
            raise ValueError(
                f"{joint_name} hierarchy check failed: fixed={fixed_delta}, moving={moving_delta}"
            )
        motion_checks[joint_name] = {
            "upstream_displacement_m": fixed_delta,
            "downstream_change": moving_delta,
            "change_kind": "rotation_matrix" if rotation_only else "position_m",
        }
        if joint_name == "J4_END_EFFECTOR":
            mirror_delta = math.sqrt(
                sum(
                    float(
                        data.xmat[body_ids["gripper_mirror"]][index]
                        - home_rotations["gripper_mirror"][index]
                    )
                    ** 2
                    for index in range(9)
                )
            )
            if mirror_delta < 1e-5:
                raise ValueError("J4 mirror gear did not counter-rotate")
            motion_checks[joint_name]["mirror_rotation_matrix_change"] = mirror_delta
            motion_checks[joint_name]["gear_ratio"] = coupling_ratio

    data.qpos[:] = 0
    data.qvel[:] = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if any(not math.isfinite(float(value)) for value in data.qpos):
        raise ValueError("MuJoCo state became non-finite")
    return {
        "model": str(model_path).replace("\\", "/"),
        "mujoco_version": mujoco.__version__,
        "topology": actual,
        "steps": steps,
        "finite_state": True,
        "motion_checks": motion_checks,
        "scope": "kinematic_validation_only",
    }


def record_validation(
    robot_model_path: Path,
    report_path: Path,
    report: dict[str, object],
    *,
    visual_reviewed: bool,
) -> None:
    robot_model = json.loads(robot_model_path.read_text(encoding="utf-8"))
    removable = {
        "the 3mf contains millimetre print-plate layouts rather than assembled link transforms",
        "collision meshes and a mujoco model have not been validated for this assembly",
        "joint frames require parent-link transform validation",
    }
    if visual_reviewed:
        removable.add("combined collision meshes require visual validation")
    robot_model["blockers"] = [
        item for item in robot_model.get("blockers", []) if str(item).casefold() not in removable
    ]
    physical_blocker = (
        "physical dimensions, collision clearances, masses, inertia, friction, and "
        "actuator dynamics require validation before physics training"
    )
    if physical_blocker not in robot_model["blockers"]:
        robot_model["blockers"].append(physical_blocker)
    robot_model["mujoco_validation"] = {
        "model": str(Path(str(report["model"])).name),
        "report": str(report_path.name),
        "scope": report["scope"],
        "mujoco_version": report["mujoco_version"],
        "steps": report["steps"],
        "finite_state": report["finite_state"],
        "visual_reviewed": visual_reviewed,
    }
    robot_model["kinematics_validated"] = True
    robot_model_path.write_text(json.dumps(robot_model, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--robot-model", type=Path)
    parser.add_argument("--visual-reviewed", action="store_true")
    args = parser.parse_args()
    report = validate(args.model, steps=args.steps)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    if args.robot_model:
        if not args.output:
            raise ValueError("--robot-model requires --output so validation evidence is saved")
        record_validation(
            args.robot_model,
            args.output,
            report,
            visual_reviewed=args.visual_reviewed,
        )
        print(f"Updated {args.robot_model.resolve()}")
    print(payload)


if __name__ == "__main__":
    main()
