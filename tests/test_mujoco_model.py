from pathlib import Path
from xml.etree import ElementTree as ET

from tools.generate_mujoco_model import generate

MODEL = Path("robot_models/four_dof_desktop_arm/model.json")


def test_generated_mujoco_model_preserves_cad_topology_without_invented_limits(tmp_path):
    output = tmp_path / "kinematic_validation.xml"
    generate(MODEL, output)
    root = ET.parse(output).getroot()

    assert root.find("option").attrib["gravity"] == "0 0 0"
    assert {item.attrib["name"] for item in root.findall("asset/mesh")} == {
        "fixed_base",
        "turntable",
        "upper_arm",
        "forearm",
        "gripper_fixed",
        "gripper_primary",
        "gripper_mirror",
    }
    joints = root.findall("worldbody//joint")
    assert [item.attrib["name"] for item in joints] == [
        "J1_BASE_YAW",
        "J2_SHOULDER",
        "J3_ELBOW",
        "J4_END_EFFECTOR",
        "J4_END_EFFECTOR_MIRROR",
    ]
    assert all(item.attrib["limited"] == "false" for item in joints)
    coupling = root.find("equality/joint")
    assert coupling is not None
    assert coupling.attrib == {
        "name": "J4_GEAR_COUPLING",
        "joint1": "J4_END_EFFECTOR",
        "joint2": "J4_END_EFFECTOR_MIRROR",
        "polycoef": "0 -1 0 0 0",
    }
    assert root.find("actuator") is None
