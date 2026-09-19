"""Merge a Fusion robot export into MoIRA's validated arm-model metadata."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


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
    merged_meshes = {}
    destination_names: set[str] = set()
    for component, relative in exported["meshes"].items():
        source = (export_path.parent / relative).resolve()
        if not source.is_file() or export_path.parent.resolve() not in source.parents:
            raise ValueError(f"Invalid exported mesh path for {component}")
        destination = mesh_output / source.name
        if destination.name.casefold() in destination_names:
            raise ValueError(f"Exported meshes contain a duplicate filename: {destination.name}")
        destination_names.add(destination.name.casefold())
        shutil.copy2(source, destination)
        merged_meshes[component] = str(destination.relative_to(output_dir)).replace("\\", "/")

    exported_joints = exported["joints"]
    for configured in model["joints"]:
        item = _match_joint(exported_joints, configured["name"])
        transform = item.get("transform") or item.get("geometry_one_transform")
        if not transform or not transform.get("translation_m"):
            raise ValueError(f"Exported joint {item['name']} does not contain an origin")
        configured["origin_m"] = transform["translation_m"]
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

    model["geometry"] = {
        "upper_arm_m": _parameter_m(exported["parameters"], "upper_length"),
        "forearm_m": _parameter_m(exported["parameters"], "forearm_length"),
        "shoulder_height_m": _parameter_m(exported["parameters"], "shoulder_height"),
    }
    model["meshes"] = merged_meshes
    model["mesh_scale_to_m"] = exported.get("mesh_scale_to_m")
    model["blockers"] = [
        item
        for item in model.get("blockers", [])
        if "joint origins and link lengths" not in item.casefold()
    ]
    for blocker in (
        "exported STL scale requires validation against metric occurrence bounds",
        "joint frames require parent-link transform validation",
    ):
        if blocker not in model["blockers"]:
            model["blockers"].append(blocker)
    model["fusion_export"] = str(export_path.resolve())
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
