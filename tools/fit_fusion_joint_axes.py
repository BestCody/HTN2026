"""Fit canonical revolute joints from analytic cylinders exported by Fusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three numbers")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_unit_axis(value: object) -> list[float]:
    axis = _finite_vector(value, label="cylinder axis")
    length = math.sqrt(sum(item * item for item in axis))
    if length <= 1e-12:
        raise ValueError("cylinder axis must be nonzero")
    axis = [item / length for item in axis]
    dominant = max(range(3), key=lambda index: abs(axis[index]))
    if axis[dominant] < 0:
        axis = [-item for item in axis]
    return [0.0 if abs(item) < 1e-12 else item for item in axis]


def _select_largest_cylinder(occurrence: dict[str, Any], *, joint_name: str) -> dict[str, Any]:
    features = occurrence.get("cylindrical_features")
    if not isinstance(features, list) or not features:
        raise ValueError(
            f"{joint_name} source {occurrence.get('full_path')} has no cylinder features"
        )
    ranked = sorted(features, key=lambda item: float(item.get("radius_m", -1)), reverse=True)
    selected = ranked[0]
    radius = float(selected.get("radius_m", 0))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"{joint_name} selected cylinder radius is invalid")
    if len(ranked) > 1 and math.isclose(
        radius,
        float(ranked[1].get("radius_m", 0)),
        rel_tol=0,
        abs_tol=1e-7,
    ):
        raise ValueError(
            f"{joint_name} source has multiple equally large cylinders; "
            "the feature selector is ambiguous"
        )
    return selected


def _limits() -> dict[str, object]:
    return {
        "minimum_enabled": False,
        "minimum_rad": 0.0,
        "maximum_enabled": False,
        "maximum_rad": 0.0,
        "rest_enabled": False,
        "rest_rad": 0.0,
    }


def fit(exported: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    axes = mapping.get("joint_axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("Fusion mapping does not define joint_axes")
    occurrences = exported.get("occurrences")
    if not isinstance(occurrences, list):
        raise ValueError("Fusion export does not contain occurrences")
    occurrence_by_path = {
        item.get("full_path"): item
        for item in occurrences
        if isinstance(item, dict) and isinstance(item.get("full_path"), str)
    }
    inferred = []
    evidence = {}
    for joint_name, spec in axes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{joint_name} axis specification must be an object")
        if spec.get("feature_selector") != "largest_cylindrical_radius":
            raise ValueError(f"{joint_name} uses an unsupported feature selector")
        source_path = spec.get("source_occurrence")
        source = occurrence_by_path.get(source_path)
        if source is None:
            raise ValueError(f"{joint_name} source occurrence is missing: {source_path}")
        feature = _select_largest_cylinder(source, joint_name=joint_name)
        origin = _finite_vector(feature.get("origin_m"), label=f"{joint_name} origin")
        axis = _canonical_unit_axis(feature.get("axis"))
        radius = float(feature["radius_m"])
        inferred.append(
            {
                "name": joint_name,
                "kind": "inferred_cad_revolute",
                "motion_type": "adsk::fusion::RevoluteJointMotion",
                "occurrence_one": spec.get("child_link"),
                "occurrence_two": spec.get("parent_link"),
                "axis": axis,
                "position_rad": 0.0,
                "limits": _limits(),
                "transform": {"translation_m": origin, "matrix_cm": None},
            }
        )
        evidence[joint_name] = {
            "source_occurrence": source_path,
            "feature_selector": spec["feature_selector"],
            "body_index": feature.get("body_index"),
            "face_index": feature.get("face_index"),
            "radius_m": radius,
            "origin_m": origin,
            "axis": axis,
        }

    mechanism = mapping.get("gripper_mechanism")
    if not isinstance(mechanism, dict):
        raise ValueError("Fusion mapping does not define gripper_mechanism")
    groups = [
        mechanism.get("fixed_occurrences"),
        mechanism.get("primary_occurrences"),
        mechanism.get("mirror_occurrences"),
    ]
    if any(not isinstance(group, list) or not group for group in groups):
        raise ValueError("Gripper mechanism occurrence groups must be non-empty lists")
    mechanism_paths = [path for group in groups for path in group]
    expected_paths = mapping.get("links", {}).get("end_effector")
    if (
        not isinstance(expected_paths, list)
        or len(mechanism_paths) != len(set(mechanism_paths))
        or set(mechanism_paths) != set(expected_paths)
    ):
        raise ValueError(
            "Gripper mechanism groups must partition the end_effector occurrences"
        )
    mirror_spec = mechanism.get("mirror_axis_source")
    if not isinstance(mirror_spec, dict):
        raise ValueError("Gripper mechanism does not define mirror_axis_source")
    if mirror_spec.get("feature_selector") != "largest_cylindrical_radius":
        raise ValueError("Gripper mirror uses an unsupported feature selector")
    mirror_source_path = mirror_spec.get("source_occurrence")
    mirror_source = occurrence_by_path.get(mirror_source_path)
    if mirror_source is None:
        raise ValueError(f"Gripper mirror source is missing: {mirror_source_path}")
    mirror_feature = _select_largest_cylinder(
        mirror_source, joint_name=str(mechanism.get("mirror_joint"))
    )
    gear_ratio = float(mechanism.get("gear_ratio", 0))
    if not math.isfinite(gear_ratio) or gear_ratio == 0:
        raise ValueError("Gripper mechanism gear_ratio must be finite and nonzero")
    exported["gripper_mechanism"] = {
        "primary_joint": mechanism.get("primary_joint"),
        "mirror_joint": mechanism.get("mirror_joint"),
        "gear_ratio": gear_ratio,
        "fixed_occurrences": groups[0],
        "primary_occurrences": groups[1],
        "mirror_occurrences": groups[2],
        "mirror_axis": _canonical_unit_axis(mirror_feature.get("axis")),
        "mirror_origin_m": _finite_vector(
            mirror_feature.get("origin_m"), label="gripper mirror origin"
        ),
        "mirror_evidence": {
            "source_occurrence": mirror_source_path,
            "feature_selector": mirror_spec["feature_selector"],
            "body_index": mirror_feature.get("body_index"),
            "face_index": mirror_feature.get("face_index"),
            "radius_m": float(mirror_feature["radius_m"]),
        },
    }

    existing = [
        item
        for item in exported.get("joints", [])
        if isinstance(item, dict) and item.get("name") not in axes
    ]
    exported["joints"] = existing + inferred
    capture = exported.setdefault("joint_geometry_capture", {})
    capture.update(
        {
            "method": "analytic_cylindrical_faces",
            "manual_joint_creation_required": False,
            "fitted": True,
            "evidence": evidence,
        }
    )
    geometry_complete = bool(exported.get("geometry_complete"))
    exported["complete"] = geometry_complete and set(axes) <= {
        str(item.get("name")) for item in exported["joints"]
    }
    if not exported["complete"]:
        raise ValueError("Fusion export is still incomplete after fitting joint axes")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fusion_export", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.fusion_export
    exported = json.loads(args.fusion_export.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    fitted = fit(exported, mapping)
    output.write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    print(f"Fitted {len(mapping['joint_axes'])} joint axes into {output.resolve()}")


if __name__ == "__main__":
    main()
