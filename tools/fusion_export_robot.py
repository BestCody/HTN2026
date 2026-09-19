"""Fusion 360 script that exports exact robot geometry and assembly metadata.

Run this from Fusion's Scripts and Add-Ins dialog after importing the supplied
``Sharma_Ishaan_robotAssem.SLDASM`` assembly. The Fusion-only ``adsk`` module is
intentionally imported inside ``run`` so normal repository tooling can compile
and inspect this file.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

LINK_COMPONENTS = {
    "fixed_base",
    "turntable",
    "upper_arm",
    "forearm",
    "end_effector",
}
JOINT_NAMES = {"J1_BASE_YAW", "J2_SHOULDER", "J3_ELBOW", "J4_END_EFFECTOR"}
MAPPING_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot_models"
    / "four_dof_desktop_arm"
    / "fusion_mapping.json"
)


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


def _cylindrical_features(occurrence: Any) -> list[dict[str, Any]]:
    """Return analytic cylinder faces in the occurrence's assembly context.

    Imported assemblies often lose their original mates. Cylinder axes let the
    offline model builder recover revolute-joint origins directly from the CAD
    instead of requiring the operator to recreate every joint by hand.
    """
    features: list[dict[str, Any]] = []
    for body_index in range(occurrence.bRepBodies.count):
        body = occurrence.bRepBodies.item(body_index)
        for face_index in range(body.faces.count):
            face = body.faces.item(face_index)
            geometry = face.geometry
            if not str(geometry.objectType).endswith("::Cylinder"):
                continue
            features.append(
                {
                    "body_index": body_index,
                    "face_index": face_index,
                    "origin_m": _point_m(geometry.origin),
                    "axis": _vector(geometry.axis),
                    "radius_m": float(geometry.radius) / 100,
                    "centroid_m": _point_m(face.centroid),
                    "area_m2": float(face.area) / 10_000,
                    "bounds": _bounds(face.boundingBox),
                }
            )
    return features


def _load_mapping() -> dict[str, Any]:
    value = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("links"), dict):
        raise ValueError("Fusion mapping must use schema_version 1 and contain links")
    if set(value["links"]) != LINK_COMPONENTS:
        raise ValueError("Fusion mapping link names do not match the robot model")
    for link, paths in value["links"].items():
        if (
            not isinstance(paths, list)
            or not paths
            or len(set(paths)) != len(paths)
            or any(not isinstance(path, str) or not path.strip() for path in paths)
        ):
            raise ValueError(f"Fusion mapping for {link} must list unique occurrence paths")
    return value


def run(context: Any) -> None:  # Fusion calls this entry point.
    import adsk.core  # type: ignore[import-not-found]
    import adsk.fusion  # type: ignore[import-not-found]

    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            raise RuntimeError("Import Sharma_Ishaan_robotAssem.SLDASM before running this script")

        dialog = ui.createFolderDialog()
        dialog.title = "Choose the MoIRA robot-model export directory"
        if dialog.showDialog() != adsk.core.DialogResults.DialogOK:
            return
        destination = Path(dialog.folder) / "four_dof_desktop_arm_export"
        mesh_dir = destination / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        root = design.rootComponent
        mapping = _load_mapping()
        export_manager = design.exportManager
        occurrences = []
        raw_meshes = {}
        raw_body_meshes = {}
        meshes = {}
        for occurrence in root.allOccurrences:
            component_name = occurrence.component.name
            occurrence_path = _occurrence_name(occurrence)
            record = {
                "name": occurrence.name,
                "full_path": _occurrence_name(occurrence),
                "component": component_name,
                "transform": _matrix(occurrence.transform2),
                "bounds": _bounds(occurrence.preciseBoundingBox),
                "physical_properties": _physical_properties(occurrence),
                "cylindrical_features": _cylindrical_features(occurrence),
                "grounded": bool(occurrence.isGrounded),
            }
            occurrences.append(record)
            path = mesh_dir / f"{_safe_name(occurrence_path)}.stl"
            options = export_manager.createSTLExportOptions(occurrence, str(path))
            options.sendToPrintUtility = False
            if not export_manager.execute(options):
                raise RuntimeError(f"Fusion failed to export {occurrence_path}")
            relative = str(path.relative_to(destination)).replace("\\", "/")
            raw_meshes[occurrence_path] = relative
            body_paths = []
            for body_index in range(occurrence.bRepBodies.count):
                body = occurrence.bRepBodies.item(body_index)
                body_path = mesh_dir / (f"{_safe_name(occurrence_path)}__body_{body_index}.stl")
                body_options = export_manager.createSTLExportOptions(body, str(body_path))
                body_options.sendToPrintUtility = False
                if not export_manager.execute(body_options):
                    raise RuntimeError(
                        f"Fusion failed to export body {body_index} of {occurrence_path}"
                    )
                body_paths.append(str(body_path.relative_to(destination)).replace("\\", "/"))
            if body_paths:
                raw_body_meshes[occurrence_path] = body_paths
            if component_name in LINK_COMPONENTS:
                if component_name in meshes:
                    raise RuntimeError(
                        f"More than one occurrence uses canonical link name {component_name}"
                    )
                meshes[component_name] = relative

        mapping_issues = []
        link_meshes = {}
        if app.activeDocument.name != mapping.get("document_name"):
            mapping_issues.append(
                f"active document is {app.activeDocument.name!r}; expected "
                f"{mapping.get('document_name')!r}"
            )
        for link, paths in mapping["links"].items():
            missing = [path for path in paths if path not in raw_meshes]
            if missing:
                mapping_issues.append(f"{link} is missing: {', '.join(missing)}")
                continue
            link_meshes[link] = [raw_meshes[path] for path in paths]
            # Retained for compatibility with older merge tools. The current
            # merge uses every entry in link_meshes to create one link-local mesh.
            meshes[link] = link_meshes[link][0]

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
            "link_meshes": link_meshes,
            "link_occurrences": mapping["links"],
            "raw_meshes": raw_meshes,
            "raw_body_meshes": raw_body_meshes,
            "coordinate_frame": mapping["coordinate_frame"],
            "joint_chain": mapping["joint_chain"],
            "mapping_issues": mapping_issues,
            "joint_geometry_capture": {
                "method": "analytic_cylindrical_faces",
                "manual_joint_creation_required": False,
                "coordinate_space": "root_assembly",
            },
            # STL stores raw coordinates without a unit declaration. Validate the
            # resulting scale against the exported metric bounds before collision use.
            "mesh_units": "unitless",
            "mesh_scale_to_m": None,
            "expected_link_components": sorted(LINK_COMPONENTS),
            "expected_joint_names": sorted(JOINT_NAMES),
            "geometry_complete": not mapping_issues and LINK_COMPONENTS == set(link_meshes),
            "complete": not mapping_issues
            and LINK_COMPONENTS == set(link_meshes)
            and all(any(item["name"].startswith(name) for item in joints) for name in JOINT_NAMES),
        }
        output_path = destination / "fusion_robot_export.json"
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        ui.messageBox(
            f"Exported {len(raw_meshes)} occurrence meshes, "
            f"{sum(len(paths) for paths in raw_body_meshes.values())} body meshes, "
            f"{len(link_meshes)} mapped links, {len(joints)} existing joints, "
            f"and analytic cylinder axes to:\n{destination}\n\n"
            "You do not need to create Fusion joints manually."
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
