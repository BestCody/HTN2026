import hashlib
import json
from pathlib import Path

from moira.robot_config import load_robot_model
from moira.robot_sources import inspect_3mf

MODEL_DIR = Path("robot_models/four_dof_desktop_arm")
MODEL_PATH = MODEL_DIR / "model.json"


def test_new_robot_source_manifest_is_complete_and_valid():
    manifest = json.loads((MODEL_DIR / "source_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 18
    assert {item["purpose"] for item in artifacts} == {
        "assembly_source",
        "cad_dependency",
        "print_plate_meshes",
    }
    for item in artifacts:
        target = (MODEL_DIR / item["path"]).resolve()
        assert target.is_file(), item["path"]
        assert target.stat().st_size == item["bytes"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]

    load_robot_model(MODEL_PATH, verify_source=True)


def test_3mf_reports_print_geometry_without_claiming_assembly_kinematics():
    report = inspect_3mf("Robotic+Arm.3mf")
    assert report.unit == "millimeter"
    assert report.plate_count == 6
    assert report.build_items == 54
    assert len(report.mesh_names) == 12
    assert report.triangle_count == 41_638
    assert report.repaired_meshes == ()
    assert not report.has_joint_metadata
    assert report.layout_kind == "print_plate"
    assert {
        "Sharma_Ishaan_base1.STL",
        "Sharma_Ishaan_base2.STL",
        "Sharma_Ishaan_rotateBasev2.STL",
        "sharma_ishaan_Link1.STL",
        "second arm.STL",
        "gripper.stp",
    } <= set(report.mesh_names)
