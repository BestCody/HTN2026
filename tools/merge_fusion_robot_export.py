"""Merge a Fusion robot export into MoIRA's validated arm-model metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any


def _parameter_m(parameters: list[dict[str, object]], name: str) -> float:
    match = next((item for item in parameters if item.get("name") == name), None)
    if match is None:
        raise ValueError(f"Fusion export is missing required parameter {name}")
    unit = str(match.get("unit", "")).strip().casefold()
    length_units = {
        "mm",
        "millimeter",
        "millimeters",
        "cm",
        "centimeter",
        "centimeters",
        "m",
        "meter",
        "meters",
        "in",
        "inch",
        "inches",
        "ft",
        "foot",
        "feet",
    }
    if unit not in length_units:
        raise ValueError(f"Fusion parameter {name} is not a supported length: {unit}")
    # Fusion's API database value for design lengths is always centimetres.
    value = float(match["database_value"]) / 100
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Fusion parameter {name} must be finite and positive")
    return value


def _match_joint(exported: list[dict[str, object]], configured_name: str) -> dict[str, object]:
    matches = [item for item in exported if str(item.get("name", "")).startswith(configured_name)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one exported joint beginning with {configured_name}, got {len(matches)}"
        )
    return matches[0]


def _joint_origin(item: dict[str, object]) -> tuple[float, float, float]:
    transform = item.get("transform") or item.get("geometry_one_transform")
    if not isinstance(transform, dict):
        raise ValueError(f"Exported joint {item.get('name')} does not contain a transform")
    translation = transform.get("translation_m")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError(f"Exported joint {item.get('name')} does not contain an origin")
    origin = tuple(float(value) for value in translation)
    if any(not math.isfinite(value) for value in origin):
        raise ValueError(f"Exported joint {item.get('name')} origin is not finite")
    return origin


def _joint_axis(item: dict[str, object]) -> tuple[float, float, float]:
    value = item.get("axis")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Exported joint {item.get('name')} does not contain an axis")
    axis = tuple(float(component) for component in value)
    length = math.sqrt(sum(component * component for component in axis))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"Exported joint {item.get('name')} axis is invalid")
    return tuple(component / length for component in axis)


def _binary_stl(path: Path) -> list[tuple[tuple[float, ...], int]]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Fusion mesh is not a binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Fusion mesh has an invalid binary STL size: {path}")
    triangles = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", raw, offset)
        triangles.append((tuple(float(value) for value in values[:12]), values[12]))
        offset += 50
    return triangles


def _rotation_and_translation(
    occurrence: dict[str, object],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    transform = occurrence.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"Occurrence {occurrence.get('full_path')} has no transform")
    matrix = transform.get("matrix_cm")
    translation = transform.get("translation_m")
    if not isinstance(matrix, list) or len(matrix) != 16:
        raise ValueError(f"Occurrence {occurrence.get('full_path')} has no 4x4 matrix")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError(f"Occurrence {occurrence.get('full_path')} has no translation")
    rotation = (
        (float(matrix[0]), float(matrix[1]), float(matrix[2])),
        (float(matrix[4]), float(matrix[5]), float(matrix[6])),
        (float(matrix[8]), float(matrix[9]), float(matrix[10])),
    )
    return rotation, tuple(float(value) for value in translation)


def _rotate(
    rotation: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in rotation)


def _world_vertex(
    rotation: tuple[tuple[float, float, float], ...],
    translation: tuple[float, float, float],
    vertex: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    rotated = _rotate(rotation, tuple(value * scale for value in vertex))
    return tuple(rotated[index] + translation[index] for index in range(3))


def _occurrence_mesh_paths(
    exported: dict[str, Any], occurrence_path: str, *, base_dir: Path | None = None
) -> list[str]:
    body_meshes = exported.get("raw_body_meshes", {}).get(occurrence_path)
    if isinstance(body_meshes, list) and body_meshes:
        if any(not isinstance(path, str) for path in body_meshes):
            raise ValueError(f"Body mesh paths for {occurrence_path} are invalid")
        existing = (
            body_meshes
            if base_dir is None
            else [path for path in body_meshes if (base_dir / path).is_file()]
        )
        if existing:
            return existing
    relative = exported["raw_meshes"][occurrence_path]
    if not isinstance(relative, str):
        raise ValueError(f"Occurrence mesh path for {occurrence_path} is invalid")
    return [relative]


def _infer_stl_scale(
    exported: dict[str, Any],
    export_path: Path,
    occurrence_by_path: dict[str, dict[str, object]],
) -> tuple[float, dict[str, float]]:
    selected_paths = {path for paths in exported["link_occurrences"].values() for path in paths}
    candidates = (0.001, 0.01, 0.0254, 1.0)
    scores: dict[float, dict[str, float]] = {}
    for scale in candidates:
        errors = {}
        for occurrence_path in selected_paths:
            occurrence = occurrence_by_path[occurrence_path]
            bounds = occurrence.get("bounds")
            if not isinstance(bounds, dict):
                raise ValueError(f"Occurrence {occurrence_path} has no metric bounds")
            expected_min, expected_max = bounds.get("min_m"), bounds.get("max_m")
            if not isinstance(expected_min, list) or not isinstance(expected_max, list):
                raise ValueError(f"Occurrence {occurrence_path} has invalid metric bounds")
            rotation, translation = _rotation_and_translation(occurrence)
            vertices = []
            for relative in _occurrence_mesh_paths(
                exported, occurrence_path, base_dir=export_path.parent
            ):
                triangles = _binary_stl((export_path.parent / relative).resolve())
                for values, _ in triangles:
                    for offset in (3, 6, 9):
                        vertices.append(
                            _world_vertex(
                                rotation,
                                translation,
                                (values[offset], values[offset + 1], values[offset + 2]),
                                scale,
                            )
                        )
            measured_min = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
            measured_max = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
            errors[occurrence_path] = max(
                *(abs(measured_min[axis] - float(expected_min[axis])) for axis in range(3)),
                *(abs(measured_max[axis] - float(expected_max[axis])) for axis in range(3)),
            )
        scores[scale] = errors

    def robust_error(candidate: float) -> float:
        values = sorted(scores[candidate].values())
        return values[max(0, math.ceil(len(values) * 0.9) - 1)]

    scale = min(scores, key=robust_error)
    tolerance_m = 0.0001
    matching = {path: error for path, error in scores[scale].items() if error <= tolerance_m}
    required = math.ceil(len(selected_paths) * 0.9)
    if len(matching) < required:
        raise ValueError(
            "Could not validate Fusion STL units against occurrence bounds; "
            f"best scale={scale}, only {len(matching)}/{len(selected_paths)} "
            "occurrences matched within 0.1 mm"
        )
    outliers = {path: error for path, error in scores[scale].items() if error > tolerance_m}
    return scale, outliers


def _write_link_mesh(
    destination: Path,
    occurrence_paths: list[str],
    *,
    exported: dict[str, Any],
    export_path: Path,
    occurrence_by_path: dict[str, dict[str, object]],
    stl_scale_to_m: float,
    link_origin_m: tuple[float, float, float],
) -> None:
    triangles: list[tuple[tuple[float, ...], int]] = []
    for occurrence_path in occurrence_paths:
        occurrence = occurrence_by_path[occurrence_path]
        rotation, translation = _rotation_and_translation(occurrence)
        for relative in _occurrence_mesh_paths(
            exported, occurrence_path, base_dir=export_path.parent
        ):
            for values, attribute in _binary_stl((export_path.parent / relative).resolve()):
                normal = _rotate(rotation, (values[0], values[1], values[2]))
                length = math.sqrt(sum(value * value for value in normal))
                if length:
                    normal = tuple(value / length for value in normal)
                transformed: list[float] = [*normal]
                for offset in (3, 6, 9):
                    world = _world_vertex(
                        rotation,
                        translation,
                        (values[offset], values[offset + 1], values[offset + 2]),
                        stl_scale_to_m,
                    )
                    transformed.extend(world[axis] - link_origin_m[axis] for axis in range(3))
                triangles.append((tuple(transformed), attribute))
    header = f"MoIRA combined link mesh: {destination.stem}".encode("ascii")[:80]
    payload = bytearray(header.ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(triangles)))
    for values, attribute in triangles:
        payload.extend(struct.pack("<12fH", *values, attribute))
    destination.write_bytes(payload)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _line_distance(
    left_origin: tuple[float, float, float],
    left_axis: tuple[float, float, float],
    right_origin: tuple[float, float, float],
    right_axis: tuple[float, float, float],
) -> float:
    """Return shortest distance between two infinite revolute axes."""
    delta = tuple(right_origin[index] - left_origin[index] for index in range(3))
    cross = (
        left_axis[1] * right_axis[2] - left_axis[2] * right_axis[1],
        left_axis[2] * right_axis[0] - left_axis[0] * right_axis[2],
        left_axis[0] * right_axis[1] - left_axis[1] * right_axis[0],
    )
    cross_length = math.sqrt(sum(value * value for value in cross))
    if cross_length <= 1e-10:
        perpendicular = (
            delta[1] * left_axis[2] - delta[2] * left_axis[1],
            delta[2] * left_axis[0] - delta[0] * left_axis[2],
            delta[0] * left_axis[1] - delta[1] * left_axis[0],
        )
        return math.sqrt(sum(value * value for value in perpendicular))
    return abs(sum(delta[index] * cross[index] for index in range(3))) / cross_length


def merge(model_path: Path, export_path: Path, output_path: Path) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    if not exported.get("complete"):
        raise ValueError("Fusion export is incomplete for the configured links and joints")
    expected_components = set(model.get("components", ()))
    if set(exported.get("meshes", {})) != expected_components:
        raise ValueError("Fusion export meshes do not exactly match configured robot components")
    if exported.get("mesh_units") not in (None, "unitless"):
        raise ValueError("Fusion STL meshes must be treated as unitless coordinates")

    output_dir = output_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_output = output_dir / "meshes"
    mesh_output.mkdir(parents=True, exist_ok=True)
    exported_joints = exported["joints"]
    joint_records = {}
    for configured in model["joints"]:
        item = _match_joint(exported_joints, configured["name"])
        joint_records[configured["name"]] = item
        configured["origin_m"] = list(_joint_origin(item))
        if item.get("axis") is not None:
            configured["axis"] = item["axis"]
        limits = item.get("limits", {})
        if (
            limits.get("minimum_enabled")
            and limits.get("maximum_enabled")
            and limits.get("rest_enabled")
        ):
            configured["lower_deg"] = float(limits["minimum_rad"]) * 180 / 3.141592653589793
            configured["upper_deg"] = float(limits["maximum_rad"]) * 180 / 3.141592653589793
            configured["home_deg"] = float(limits["rest_rad"]) * 180 / 3.141592653589793

    merged_meshes = {}
    mesh_bound_outliers: dict[str, float] = {}
    if exported.get("link_meshes") and exported.get("link_occurrences"):
        occurrences = exported.get("occurrences")
        if not isinstance(occurrences, list):
            raise ValueError("Fusion export needs occurrence transforms for grouped links")
        occurrence_by_path = {
            item["full_path"]: item
            for item in occurrences
            if isinstance(item, dict) and isinstance(item.get("full_path"), str)
        }
        expected_paths = {path for paths in exported["link_occurrences"].values() for path in paths}
        if not expected_paths <= set(occurrence_by_path):
            raise ValueError("Fusion link mapping references missing occurrence transforms")
        stl_scale_to_m, mesh_bound_outliers = _infer_stl_scale(
            exported, export_path, occurrence_by_path
        )
        joint_chain = exported.get("joint_chain")
        if not isinstance(joint_chain, dict):
            raise ValueError("Fusion export is missing the canonical joint chain")
        for component in sorted(expected_components):
            joint_name = joint_chain.get(component)
            origin = (
                (0.0, 0.0, 0.0)
                if joint_name is None
                else _joint_origin(joint_records[str(joint_name)])
            )
            destination = mesh_output / f"{component}.stl"
            _write_link_mesh(
                destination,
                exported["link_occurrences"][component],
                exported=exported,
                export_path=export_path,
                occurrence_by_path=occurrence_by_path,
                stl_scale_to_m=stl_scale_to_m,
                link_origin_m=origin,
            )
            merged_meshes[component] = str(destination.relative_to(output_dir)).replace("\\", "/")
        mechanism = exported.get("gripper_mechanism")
        if isinstance(mechanism, dict):
            primary_joint_name = str(mechanism.get("primary_joint"))
            primary_origin = _joint_origin(joint_records[primary_joint_name])
            forearm_origin = _joint_origin(joint_records["J3_ELBOW"])
            mirror_origin_raw = mechanism.get("mirror_origin_m")
            mirror_axis_raw = mechanism.get("mirror_axis")
            if not isinstance(mirror_origin_raw, list) or len(mirror_origin_raw) != 3:
                raise ValueError("Gripper mechanism mirror origin is invalid")
            if not isinstance(mirror_axis_raw, list) or len(mirror_axis_raw) != 3:
                raise ValueError("Gripper mechanism mirror axis is invalid")
            mirror_origin = tuple(float(value) for value in mirror_origin_raw)
            mechanism_meshes = {}
            mechanism_specs = (
                ("fixed", mechanism.get("fixed_occurrences"), forearm_origin),
                ("primary", mechanism.get("primary_occurrences"), primary_origin),
                ("mirror", mechanism.get("mirror_occurrences"), mirror_origin),
            )
            for name, paths, origin in mechanism_specs:
                if not isinstance(paths, list) or not paths:
                    raise ValueError(f"Gripper mechanism {name} group is invalid")
                destination = mesh_output / f"gripper_{name}.stl"
                _write_link_mesh(
                    destination,
                    paths,
                    exported=exported,
                    export_path=export_path,
                    occurrence_by_path=occurrence_by_path,
                    stl_scale_to_m=stl_scale_to_m,
                    link_origin_m=origin,
                )
                mechanism_meshes[name] = str(
                    destination.relative_to(output_dir)
                ).replace("\\", "/")
            model["gripper_mechanism"] = {
                "primary_joint": primary_joint_name,
                "mirror_joint": str(mechanism.get("mirror_joint")),
                "gear_ratio": float(mechanism.get("gear_ratio")),
                "mirror_origin_m": list(mirror_origin),
                "mirror_axis": [float(value) for value in mirror_axis_raw],
                "meshes": mechanism_meshes,
            }
        model["mesh_scale_to_m"] = 1.0
        model["fusion_mesh_validation"] = {
            "source_stl_scale_to_m": stl_scale_to_m,
            "bounds_tolerance_m": 0.0001,
            "matched_occurrences": len(expected_paths) - len(mesh_bound_outliers),
            "total_occurrences": len(expected_paths),
            "bounds_outliers_m": mesh_bound_outliers,
        }
    else:
        destination_names: set[str] = set()
        for component, relative in exported["meshes"].items():
            source = (export_path.parent / relative).resolve()
            if not source.is_file() or export_path.parent.resolve() not in source.parents:
                raise ValueError(f"Invalid exported mesh path for {component}")
            destination = mesh_output / source.name
            if destination.name.casefold() in destination_names:
                raise ValueError(
                    f"Exported meshes contain a duplicate filename: {destination.name}"
                )
            destination_names.add(destination.name.casefold())
            shutil.copy2(source, destination)
            merged_meshes[component] = str(destination.relative_to(output_dir)).replace("\\", "/")
        model["mesh_scale_to_m"] = exported.get("mesh_scale_to_m")

    parameters = exported.get("parameters", [])
    if all(
        any(item.get("name") == name for item in parameters)
        for name in ("upper_length", "forearm_length", "shoulder_height")
    ):
        geometry = {
            "upper_arm_m": _parameter_m(parameters, "upper_length"),
            "forearm_m": _parameter_m(parameters, "forearm_length"),
            "shoulder_height_m": _parameter_m(parameters, "shoulder_height"),
        }
    else:
        j2 = _joint_origin(joint_records["J2_SHOULDER"])
        j3 = _joint_origin(joint_records["J3_ELBOW"])
        j4 = _joint_origin(joint_records["J4_END_EFFECTOR"])
        coordinate_frame = exported.get("coordinate_frame")
        if not isinstance(coordinate_frame, dict):
            raise ValueError("Fusion export needs a coordinate frame to derive link geometry")
        axis_index = {"x": 0, "y": 1, "z": 2}.get(
            str(coordinate_frame.get("up_axis", "")).casefold()
        )
        if axis_index is None:
            raise ValueError("Fusion export coordinate-frame up axis is invalid")
        geometry = {
            "upper_arm_m": _line_distance(
                j2,
                _joint_axis(joint_records["J2_SHOULDER"]),
                j3,
                _joint_axis(joint_records["J3_ELBOW"]),
            ),
            "forearm_m": _line_distance(
                j3,
                _joint_axis(joint_records["J3_ELBOW"]),
                j4,
                _joint_axis(joint_records["J4_END_EFFECTOR"]),
            ),
            "shoulder_height_m": j2[axis_index],
        }
        model["coordinate_frame"] = coordinate_frame
    model["geometry"] = geometry
    model["meshes"] = merged_meshes
    model["kinematics_validated"] = False
    model["collision_geometry_validated"] = False
    model.pop("mujoco_validation", None)
    model["blockers"] = [
        item
        for item in model.get("blockers", [])
        if not ("joint" in item.casefold() and "assembly export" in item.casefold())
    ]
    remaining = ["joint frames require parent-link transform validation"]
    if not exported.get("link_meshes"):
        remaining.append("exported STL scale requires validation against metric occurrence bounds")
    else:
        remaining.append("combined collision meshes require visual validation")
        if mesh_bound_outliers:
            remaining.append(
                "Fusion STL bounds exclude non-meshable or hidden CAD geometry: "
                + ", ".join(sorted(mesh_bound_outliers))
            )
    for blocker in remaining:
        if blocker not in model["blockers"]:
            model["blockers"].append(blocker)
    model["fusion_export"] = os.path.relpath(
        export_path.resolve(), output_path.parent.resolve()
    ).replace("\\", "/")
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("fusion_export", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = merge(args.model, args.fusion_export, args.output)
    print(f"Wrote {args.output.resolve()}")
    print(f"Remaining readiness blockers: {len(model.get('blockers', []))}")


if __name__ == "__main__":
    main()
