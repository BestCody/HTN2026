from pathlib import Path

import pytest

from moira.physical import SimulationOutcome, _wire_equivalent
from moira.software_integration import ImageFileCamera, load_software_integration_config
from moira.specialists import PlanarBimanualIK


def test_software_profile_is_explicitly_nonphysical():
    config = load_software_integration_config("config/software_integration.json")
    assert config.profile.simulation_horizon_seconds == 2.5
    assert config.robot_model.name == "model.json"
    assert config.assumptions["per_arm_payload_kg"] == 0.025


def test_image_file_camera_rejects_empty_fixture(tmp_path: Path):
    image = tmp_path / "empty.jpg"
    image.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        ImageFileCamera(image).capture()


def test_json_round_trip_sequences_remain_strictly_equivalent():
    original = SimulationOutcome("p", 0.8, True, 2.5, (), {"models": ("a", "b")})
    decoded = SimulationOutcome("p", 0.8, True, 2.5, (), {"models": ["a", "b"]})
    assert original != decoded
    assert _wire_equivalent(original) == _wire_equivalent(decoded)


def test_single_arm_ik_allows_zero_bimanual_mount_offset():
    solver = PlanarBimanualIK(shoulder_offset_m=0.0)
    assert solver.shoulder_offset_m == 0.0
