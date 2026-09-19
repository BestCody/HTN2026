import importlib.util
import json
from pathlib import Path

import pytest

from tools.fit_fusion_joint_axes import fit
from tools.merge_fusion_robot_export import _line_distance, _occurrence_mesh_paths

BUNDLE = Path("tools/FusionExportRobot")


def test_fusion_export_bundle_has_matching_entrypoint_and_manifest():
    manifest = json.loads((BUNDLE / "FusionExportRobot.manifest").read_text(encoding="utf-8"))

    assert manifest["autodeskProduct"] == "Fusion360"
    assert manifest["type"] == "script"
    assert manifest["supportedOS"] == "windows|mac"
    assert (BUNDLE / "FusionExportRobot.py").is_file()


def test_fusion_export_bundle_loads_maintained_implementation():
    spec = importlib.util.spec_from_file_location(
        "test_fusion_export_entrypoint", BUNDLE / "FusionExportRobot.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    implementation = module._implementation()
    assert callable(implementation.run)
    assert callable(implementation.stop)
    assert implementation.LINK_COMPONENTS == {
        "fixed_base",
        "turntable",
        "upper_arm",
        "forearm",
        "end_effector",
    }
    assert implementation.JOINT_NAMES == {
        "J1_BASE_YAW",
        "J2_SHOULDER",
        "J3_ELBOW",
        "J4_END_EFFECTOR",
    }
    assert callable(implementation._cylindrical_features)


def test_fusion_mapping_groups_original_occurrences_without_duplicates():
    mapping = json.loads(
        Path("robot_models/four_dof_desktop_arm/fusion_mapping.json").read_text(encoding="utf-8")
    )
    paths = [path for values in mapping["links"].values() for path in values]
    assert mapping["schema_version"] == 1
    assert set(mapping["links"]) == {
        "fixed_base",
        "turntable",
        "upper_arm",
        "forearm",
        "end_effector",
    }
    assert len(paths) == len(set(paths)) == 37
    assert set(mapping["joint_chain"].values()) == {
        "J1_BASE_YAW",
        "J2_SHOULDER",
        "J3_ELBOW",
        "J4_END_EFFECTOR",
    }
    assert set(mapping["joint_axes"]) == set(mapping["joint_chain"].values())
    mechanism = mapping["gripper_mechanism"]
    grouped = (
        mechanism["fixed_occurrences"]
        + mechanism["primary_occurrences"]
        + mechanism["mirror_occurrences"]
    )
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(mapping["links"]["end_effector"])


def _test_gripper_mapping():
    return {
        "links": {"end_effector": ["fixed", "pivot", "mirror"]},
        "gripper_mechanism": {
            "primary_joint": "J1",
            "mirror_joint": "J1_MIRROR",
            "gear_ratio": -1,
            "mirror_axis_source": {
                "source_occurrence": "mirror",
                "feature_selector": "largest_cylindrical_radius",
            },
            "fixed_occurrences": ["fixed"],
            "primary_occurrences": ["pivot"],
            "mirror_occurrences": ["mirror"],
        },
    }


def test_joint_axis_fit_uses_named_cad_features_without_coordinate_constants():
    exported = {
        "geometry_complete": True,
        "joints": [],
        "occurrences": [
            {
                "full_path": "pivot",
                "cylindrical_features": [
                    {
                        "body_index": 0,
                        "face_index": 1,
                        "radius_m": 0.005,
                        "origin_m": [0.1, 0.2, 0.3],
                        "axis": [0.0, 0.0, -2.0],
                    },
                    {
                        "body_index": 0,
                        "face_index": 2,
                        "radius_m": 0.003,
                        "origin_m": [9.0, 9.0, 9.0],
                        "axis": [1.0, 0.0, 0.0],
                    },
                ],
            },
            {
                "full_path": "mirror",
                "cylindrical_features": [
                    {
                        "body_index": 0,
                        "face_index": 3,
                        "radius_m": 0.004,
                        "origin_m": [0.4, 0.5, 0.6],
                        "axis": [0.0, 0.0, 1.0],
                    }
                ],
            },
        ],
    }
    mapping = _test_gripper_mapping() | {
        "joint_axes": {
            "J1": {
                "parent_link": "base",
                "child_link": "arm",
                "source_occurrence": "pivot",
                "feature_selector": "largest_cylindrical_radius",
            }
        }
    }

    fitted = fit(exported, mapping)

    assert fitted["complete"] is True
    assert fitted["joints"][0]["axis"] == [0.0, 0.0, 1.0]
    assert fitted["joints"][0]["transform"]["translation_m"] == [0.1, 0.2, 0.3]


def test_joint_axis_fit_rejects_ambiguous_equal_radius_features():
    exported = {
        "geometry_complete": True,
        "joints": [],
        "occurrences": [
            {
                "full_path": "pivot",
                "cylindrical_features": [
                    {"radius_m": 0.005, "origin_m": [0, 0, 0], "axis": [0, 0, 1]},
                    {"radius_m": 0.005, "origin_m": [1, 0, 0], "axis": [0, 0, 1]},
                ],
            },
            {
                "full_path": "mirror",
                "cylindrical_features": [
                    {"radius_m": 0.004, "origin_m": [0, 1, 0], "axis": [0, 0, 1]}
                ],
            },
        ],
    }
    mapping = _test_gripper_mapping() | {
        "joint_axes": {
            "J1": {
                "parent_link": "base",
                "child_link": "arm",
                "source_occurrence": "pivot",
                "feature_selector": "largest_cylindrical_radius",
            }
        }
    }

    with pytest.raises(ValueError, match="ambiguous"):
        fit(exported, mapping)


def test_merge_prefers_complete_body_mesh_exports_and_measures_axis_distance():
    exported = {
        "raw_meshes": {"part": "whole.stl"},
        "raw_body_meshes": {"part": ["body0.stl", "body1.stl"]},
    }
    assert _occurrence_mesh_paths(exported, "part") == ["body0.stl", "body1.stl"]
    assert _line_distance((0, 0, 0), (0, 0, 1), (3, 4, 9), (0, 0, 1)) == 5
