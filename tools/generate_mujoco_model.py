"""Generate a MuJoCo kinematic-validation model from a merged MoIRA robot model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

LEGACY_LAYOUT = "yaw_shoulder_elbow"
FIXED_ELBOW_LAYOUT = "yaw_shoulder_fixed_link"
LEGACY_JOINTS = {
    "J1_BASE_YAW",
    "J2_SHOULDER",
    "J3_ELBOW",
    "J4_END_EFFECTOR",
}
FIXED_ELBOW_JOINTS = {
    "J1_BASE_YAW",
    "J2_SHOULDER",
    "J3_GRIPPER",
}
CANONICAL_MESHES = {"fixed_base", "turntable", "upper_arm", "forearm", "end_effector"}
MECHANISM_MESHES = {"fixed", "primary", "mirror"}


def _vector(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _values(value: tuple[float, ...]) -> str:
    return " ".join(f"{item:.12g}" for item in value)


def _layout(model: dict[str, Any]) -> str:
    raw = model.get("kinematics", {})
    if raw is None:
        return LEGACY_LAYOUT
    if not isinstance(raw, dict):
        raise ValueError("Robot model kinematics must be an object")
    layout = str(raw.get("layout", LEGACY_LAYOUT))
    if layout not in {LEGACY_LAYOUT, FIXED_ELBOW_LAYOUT}:
        raise ValueError(f"Unsupported kinematic layout: {layout}")
    return layout


def _joint_by_name(
    model: dict[str, Any], expected: set[str]
) -> dict[str, dict[str, Any]]:
    joints = model.get("joints")
    if not isinstance(joints, list):
        raise ValueError("Robot model does not contain joints")
    result = {
        str(item.get("name")): item
        for item in joints
        if isinstance(item, dict) and item.get("name")
    }
    if set(result) != expected:
        raise ValueError(
            f"Robot model joints do not match the {model.get('model_id')} topology"
        )
    return result


def _fixed_elbow(model: dict[str, Any]) -> dict[str, Any]:
    kinematics = model.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("Fixed-link model is missing kinematics")
    elbow = kinematics.get("fixed_elbow")
    if not isinstance(elbow, dict) or elbow.get("rigid") is not True:
        raise ValueError("Fixed-link model must declare a rigid fixed_elbow")
    if float(elbow.get("reference_angle_deg", math.nan)) != 0.0:
        raise ValueError(
            "MuJoCo export currently requires the fixed elbow at the CAD zero pose"
        )
    return elbow


def _mesh_path(model_path: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Missing merged mesh for {label}")
    root = model_path.parent.resolve()
    path = (root / relative).resolve()
    if not path.is_file() or (path != root and root not in path.parents):
        raise ValueError(f"Invalid merged mesh for {label}: {path}")
    return path


def _joint_attributes(name: str, joint: dict[str, Any]) -> dict[str, str]:
    result = {
        "name": name,
        "type": "hinge",
        "axis": _values(_vector(joint.get("axis"), f"{name} axis")),
    }
    lower, upper = joint.get("lower_deg"), joint.get("upper_deg")
    if lower is None or upper is None:
        result["limited"] = "false"
    else:
        result["range"] = _values(
            (math.radians(float(lower)), math.radians(float(upper)))
        )
    return result


def generate(model_path: Path, output_path: Path) -> ET.ElementTree:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    layout = _layout(model)
    meshes = model.get("meshes")
    if not isinstance(meshes, dict) or set(meshes) != CANONICAL_MESHES:
        raise ValueError("Robot model must contain all five merged link meshes")
    mechanism = model.get("gripper_mechanism")
    if not isinstance(mechanism, dict):
        raise ValueError("Robot model does not contain the coupled gripper mechanism")
    mechanism_meshes = mechanism.get("meshes")
    if not isinstance(mechanism_meshes, dict) or set(mechanism_meshes) != MECHANISM_MESHES:
        raise ValueError("Robot model must contain all three gripper mechanism meshes")
    coordinate_frame = model.get("coordinate_frame")
    if coordinate_frame != {"ground_plane": "XZ", "up_axis": "Y"}:
        raise ValueError("MuJoCo generator currently requires the exported XZ/Y-up frame")
    expected_joints = (
        LEGACY_JOINTS if layout == LEGACY_LAYOUT else FIXED_ELBOW_JOINTS
    )
    joints = _joint_by_name(model, expected_joints)
    primary_name = str(mechanism.get("primary_joint"))
    mirror_name = str(mechanism.get("mirror_joint"))
    expected_primary = (
        "J4_END_EFFECTOR" if layout == LEGACY_LAYOUT else "J3_GRIPPER"
    )
    if primary_name != expected_primary or not mirror_name:
        raise ValueError("Gripper mechanism joint names do not match the robot topology")
    gear_ratio = float(mechanism.get("gear_ratio", 0))
    if not math.isfinite(gear_ratio) or gear_ratio == 0:
        raise ValueError("Gripper gear ratio must be finite and nonzero")

    root = ET.Element("mujoco", {"model": str(model.get("model_id", "moira-robot"))})
    topology_note = (
        "The physical elbow is rigid at the exported CAD assembly pose. "
        if layout == FIXED_ELBOW_LAYOUT
        else ""
    )
    root.append(
        ET.Comment(
            "Kinematic validation only: zero gravity, no actuators, and no invented "
            f"joint limits or dynamics. {topology_note}"
        )
    )
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": "meshes",
            "autolimits": "true",
            "balanceinertia": "true",
        },
    )
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    asset = ET.SubElement(root, "asset")
    asset_specs = {
        component: meshes[component]
        for component in ("fixed_base", "turntable", "upper_arm", "forearm")
    }
    asset_specs.update(
        {f"gripper_{name}": mechanism_meshes[name] for name in sorted(MECHANISM_MESHES)}
    )
    for name, relative in asset_specs.items():
        path = _mesh_path(model_path, relative, name)
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": name,
                "file": path.name,
                "scale": "1 1 1",
                "inertia": "convex",
            },
        )

    worldbody = ET.SubElement(root, "worldbody")
    bodies: dict[str, ET.Element] = {}
    world_origins: dict[str, tuple[float, float, float]] = {
        "fixed_base": (0.0, 0.0, 0.0)
    }
    fixed_elbow = _fixed_elbow(model) if layout == FIXED_ELBOW_LAYOUT else None
    chain = (
        ("fixed_base", None, None, None),
        ("turntable", "fixed_base", "J1_BASE_YAW", None),
        ("upper_arm", "turntable", "J2_SHOULDER", None),
        (
            "forearm",
            "upper_arm",
            "J3_ELBOW" if layout == LEGACY_LAYOUT else None,
            fixed_elbow,
        ),
    )
    for component, parent, joint_name, fixed_joint in chain:
        if parent is None:
            body = ET.SubElement(worldbody, "body", {"name": component})
        else:
            frame = joints[str(joint_name)] if joint_name else fixed_joint
            if not isinstance(frame, dict):
                raise ValueError(f"Missing transform for {component}")
            frame_name = str(joint_name or "fixed elbow")
            world_origin = _vector(frame.get("origin_m"), f"{frame_name} origin")
            parent_origin = world_origins[parent]
            relative = tuple(world_origin[index] - parent_origin[index] for index in range(3))
            body = ET.SubElement(
                bodies[parent], "body", {"name": component, "pos": _values(relative)}
            )
            if joint_name:
                ET.SubElement(
                    body,
                    "joint",
                    _joint_attributes(str(joint_name), joints[str(joint_name)]),
                )
            world_origins[component] = world_origin
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{component}_mesh",
                "type": "mesh",
                "mesh": component,
                "rgba": "0.55 0.62 0.72 1",
            },
        )
        bodies[component] = body

    forearm = bodies["forearm"]
    ET.SubElement(
        forearm,
        "geom",
        {
            "name": "gripper_fixed_mesh",
            "type": "mesh",
            "mesh": "gripper_fixed",
            "rgba": "0.48 0.54 0.64 1",
        },
    )
    forearm_origin = world_origins["forearm"]
    primary = joints[primary_name]
    primary_origin = _vector(primary.get("origin_m"), f"{primary_name} origin")
    primary_relative = tuple(
        primary_origin[index] - forearm_origin[index] for index in range(3)
    )
    primary_body = ET.SubElement(
        forearm,
        "body",
        {"name": "gripper_primary", "pos": _values(primary_relative)},
    )
    ET.SubElement(primary_body, "joint", _joint_attributes(primary_name, primary))
    ET.SubElement(
        primary_body,
        "geom",
        {
            "name": "gripper_primary_mesh",
            "type": "mesh",
            "mesh": "gripper_primary",
            "rgba": "0.26 0.55 0.82 1",
        },
    )

    mirror_origin = _vector(mechanism.get("mirror_origin_m"), "gripper mirror origin")
    mirror_relative = tuple(
        mirror_origin[index] - forearm_origin[index] for index in range(3)
    )
    mirror_body = ET.SubElement(
        forearm,
        "body",
        {"name": "gripper_mirror", "pos": _values(mirror_relative)},
    )
    mirror_joint = {
        "axis": list(_vector(mechanism.get("mirror_axis"), "gripper mirror axis")),
        "lower_deg": None,
        "upper_deg": None,
    }
    ET.SubElement(mirror_body, "joint", _joint_attributes(mirror_name, mirror_joint))
    ET.SubElement(
        mirror_body,
        "geom",
        {
            "name": "gripper_mirror_mesh",
            "type": "mesh",
            "mesh": "gripper_mirror",
            "rgba": "0.26 0.55 0.82 1",
        },
    )

    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "name": (
                "J4_GEAR_COUPLING"
                if layout == LEGACY_LAYOUT
                else "J3_GRIPPER_GEAR_COUPLING"
            ),
            "joint1": primary_name,
            "joint2": mirror_name,
            "polycoef": _values((0.0, gear_ratio, 0.0, 0.0, 0.0)),
        },
    )

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.model, args.output)
    print(f"Wrote kinematic-validation MJCF to {args.output.resolve()}")


if __name__ == "__main__":
    main()
