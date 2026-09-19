import importlib.util
import json
from pathlib import Path

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
