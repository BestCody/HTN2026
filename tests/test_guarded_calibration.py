from dataclasses import replace

import pytest

from moira.guarded_calibration import load_guarded_calibration_envelope


def test_guarded_envelope_matches_supplied_arduino_calibration() -> None:
    envelope = load_guarded_calibration_envelope("config/arm1_guarded_envelope.json")

    assert envelope.servos["J1_BASE_YAW"].channel == 0
    assert envelope.servos["J2_SHOULDER"].channel == 1
    assert envelope.servos["J3_GRIPPER"].channel == 2
    assert envelope.disabled_channels == (3,)
    assert envelope.command_to_tick("J1_BASE_YAW", 0) == 102
    assert envelope.command_to_tick("J1_BASE_YAW", 60) == 238
    assert envelope.command_to_tick("J1_BASE_YAW", 123) == 382
    assert envelope.command_to_tick("J2_SHOULDER", 0) == 102
    assert envelope.command_to_tick("J2_SHOULDER", 90) == 307
    assert envelope.command_to_tick("J3_GRIPPER", 83) == 291
    assert envelope.command_to_tick("J3_GRIPPER", 180) == 512


def test_guarded_ramp_rejects_range_and_rate_violations() -> None:
    envelope = load_guarded_calibration_envelope("config/arm1_guarded_envelope.json")

    with pytest.raises(ValueError, match="outside"):
        envelope.ramp("J2_SHOULDER", 0, 91, 2)
    with pytest.raises(ValueError, match="exceeds"):
        envelope.ramp("J2_SHOULDER", 0, 5, 6)
    assert envelope.ramp("J2_SHOULDER", 2, 0, 2) == (
        (2, 106),
        (1, 104),
        (0, 102),
    )


def test_guarded_envelope_cannot_enable_unused_channel() -> None:
    envelope = load_guarded_calibration_envelope("config/arm1_guarded_envelope.json")

    with pytest.raises(ValueError, match="channel 3"):
        replace(envelope, disabled_channels=())
