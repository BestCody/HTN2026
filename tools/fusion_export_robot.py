"""Fusion 360 script that exports exact robot geometry and assembly metadata.

Run this from Fusion's Scripts and Add-Ins dialog while
``Current_Lightweight_Arm.f3d`` is the active design. The Fusion-only ``adsk``
module is intentionally imported inside ``run`` so normal repository tooling can
compile and inspect this file.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

LINK_COMPONENTS = {
    "01_FIXED_BASE",
    "02_YAW_TURRET",
    "03_UPPER_LINK",
    "04_FOREARM_HEAD",
    "05_MOVING_FINGER",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.asArray()]


def _point_m(value: Any) -> list[float] | None:
    if value is None:
        return None
    # Fusion design database length units are centimetres.
    return [float(value.x) / 100, float(value.y) / 100, float(value.z) / 100]


def _matrix(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "matrix_cm": [float(item) for item in value.asArray()],
        "translation_m": [float(item) / 100 for item in value.translation.asArray()],
    }


def _bounds(value: Any) -> dict[str, list[float]] | None:
    if value is None:
        return None
    return {"min_m": _point_m(value.minPoint), "max_m": _point_m(value.maxPoint)}


def _occurrence_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "fullPathName", None) or value.name)


def _limits(value: Any) -> dict[str, Any]:
    return {
        "minimum_enabled": bool(value.isMinimumValueEnabled),
        "minimum_rad": float(value.minimumValue),
        "maximum_enabled": bool(value.isMaximumValueEnabled),
        "maximum_rad": float(value.maximumValue),
        "rest_enabled": bool(value.isRestValueEnabled),
        "rest_rad": float(value.restValue),
    }


def _joint(value: Any, *, as_built: bool) -> dict[str, Any]:
    motion = value.jointMotion
    result = {
        "name": value.name,
        "kind": "as_built" if as_built else "joint",
        "motion_type": motion.objectType,
        "occurrence_one": _occurrence_name(value.occurrenceOne),
        "occurrence_two": _occurrence_name(value.occurrenceTwo),
    }
    if as_built:
        result["transform"] = _matrix(value.transform)
    else:
        result["geometry_one_transform"] = _matrix(value.geometryOneTransform)
        result["geometry_two_transform"] = _matrix(value.geometryTwoTransform)
    if "RevoluteJointMotion" in motion.objectType:
        result.update(
            {
                "axis": _vector(motion.rotationAxisVector),
                "position_rad": float(motion.rotationValue),
                "limits": _limits(motion.rotationLimits),
            }
        )
    return result


def _physical_properties(occurrence: Any) -> dict[str, Any]:
    properties = occurrence.getPhysicalProperties()
    return {
        "mass_kg": float(properties.mass),
        "volume_m3": float(properties.volume) / 1_000_000,
        "area_m2": float(properties.area) / 10_000,
        "center_of_mass_m": _point_m(properties.centerOfMass),
    }


def run(context: Any) -> None:  # Fusion calls this entry point.
    import adsk.core  # type: ignore[import-not-found]
    import adsk.fusion  # type: ignore[import-not-found]

    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            raise RuntimeError("Open Current_Lightweight_Arm.f3d before running this script")

        dialog = ui.createFolderDialog()
        dialog.title = "Choose the MoIRA robot-model export directory"
        if dialog.showDialog() != adsk.core.DialogResults.DialogOK:
            return
        destination = Path(dialog.folder) / "current_lightweight_arm_export"
        mesh_dir = destination / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        root = design.rootComponent
        export_manager = design.exportManager
        occurrences = []
        meshes = {}
        for occurrence in root.allOccurrences:
            component_name = occurrence.component.name
            record = {
                "name": occurrence.name,
                "full_path": _occurrence_name(occurrence),
                "component": component_name,
                "transform": _matrix(occurrence.transform2),
                "bounds": _bounds(occurrence.preciseBoundingBox),
                "physical_properties": _physical_properties(occurrence),
                "grounded": bool(occurrence.isGrounded),
            }
            occurrences.append(record)
            if component_name in LINK_COMPONENTS and component_name not in meshes:
                path = mesh_dir / f"{_safe_name(component_name)}.stl"
                options = export_manager.createSTLExportOptions(occurrence, str(path))
                options.sendToPrintUtility = False
                if not export_manager.execute(options):
                    raise RuntimeError(f"Fusion failed to export {component_name}")
                meshes[component_name] = str(path.relative_to(destination)).replace("\\", "/")

        parameters = []
        for parameter in design.userParameters:
            parameters.append(
                {
                    "name": parameter.name,
                    "expression": parameter.expression,
                    "unit": parameter.unit,
                    "database_value": float(parameter.value),
                    "comment": parameter.comment,
                }
            )

        joints = [_joint(item, as_built=False) for item in root.allJoints]
        joints.extend(_joint(item, as_built=True) for item in root.allAsBuiltJoints)
        output = {
            "schema_version": 1,
            "document_name": app.activeDocument.name,
            "database_units": {"length": "cm", "angle": "rad", "mass": "kg"},
            "parameters": parameters,
            "occurrences": occurrences,
            "joints": joints,
            "meshes": meshes,
            # STL stores raw coordinates without a unit declaration. Validate the
            # resulting scale against the exported metric bounds before collision use.
            "mesh_units": "unitless",
            "mesh_scale_to_m": None,
            "expected_link_components": sorted(LINK_COMPONENTS),
            "complete": LINK_COMPONENTS == set(meshes) and len(joints) >= 4,
        }
        output_path = destination / "fusion_robot_export.json"
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        ui.messageBox(
            f"Exported {len(meshes)} link meshes and {len(joints)} joints to:\n{destination}"
        )
    except Exception as exc:
        message = traceback.format_exc()
        if ui is not None:
            ui.messageBox(f"Robot export failed:\n{message}")
        else:
            raise RuntimeError(message) from exc


def stop(context: Any) -> None:
    del context


if __name__ == "__main__":
    raise SystemExit("Run this script inside Autodesk Fusion, not with system Python")
