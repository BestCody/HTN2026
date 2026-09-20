from __future__ import annotations

import math
from pathlib import Path

import pytest

from moira.physical import (
    ActionChunk,
    DetectedObject,
    MotionTrajectory,
    TrajectoryPoint,
    WorldState,
)
from moira.robot_config import load_robot_model
from moira.simulation_demo import (
    MujocoArmDriver,
    MujocoCameraSource,
    MujocoDemoScene,
    load_simulation_demo_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "simulation_demo.json"


def test_simulation_demo_config_reuses_nonphysical_pipeline_contract() -> None:
    config = load_simulation_demo_config(CONFIG)

    assert config.pipeline.path == CONFIG.resolve()
    assert config.pipeline.workspace["perception"]["labels"] == [
        "red cube",
        "green platform",
    ]
    assert config.mujoco_model.name == "simulation_demo.xml"
    assert config.pipeline.robot_model.name == "physical_three_actuator_model.json"
    assert config.width == 640
    assert config.height == 480


def test_mujoco_scene_supplies_real_camera_frame_and_arm_pov(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("PIL")
    config = load_simulation_demo_config(CONFIG)
    scene = MujocoDemoScene(config.mujoco_model, width=320, height=240, fps=5)
    try:
        calibration = scene.camera_calibration()
        frame = MujocoCameraSource(scene).capture()
        scene.record_frame("TEST FRAME")
        artifacts = scene.save_artifacts(tmp_path, None)

        assert calibration["resolution_px"] == [320, 240]
        assert len(calibration["camera_to_base_matrix"]) == 4
        assert frame.camera_id == "sim-overview"
        assert frame.data.startswith(b"\xff\xd8")
        assert artifacts.animation.is_file()
        assert artifacts.final_arm_pov.is_file()
    finally:
        scene.close()


def test_demo_objects_stay_on_the_physical_fixed_link_workspace_arc() -> None:
    pytest.importorskip("mujoco")
    config = load_simulation_demo_config(CONFIG)
    model = load_robot_model(config.pipeline.robot_model)
    scene = MujocoDemoScene(config.mujoco_model, width=320, height=240, fps=5)
    try:
        assert model.effective_reach_m is not None
        tolerance = config.pipeline.assumptions["fixed_link_reach_tolerance_m"]
        for position in (scene.cube_position, scene.platform_position):
            x, y, z = position
            radial = math.hypot(x, z)
            distance = math.hypot(radial, y - model.shoulder_height_m)
            assert abs(distance - model.effective_reach_m) <= tolerance
    finally:
        scene.close()


def test_mujoco_driver_executes_routed_trajectory_and_moves_cube() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("PIL")
    config = load_simulation_demo_config(CONFIG)
    scene = MujocoDemoScene(config.mujoco_model, width=320, height=240, fps=5)
    try:
        platform = scene.platform_position
        chunk = ActionChunk(
            "candidate-chunk-transfer",
            "candidate-transfer",
            "waypoint",
            ("left",),
            0.1,
            "red-cube-1",
            (*platform, 0.0, 0.0, 0.0, 1.0),
            0.03,
            2.0,
        )
        trajectory = MotionTrajectory(
            "candidate",
            (
                TrajectoryPoint(
                    0.05,
                    {"left": (0.1, -0.2)},
                    chunk.id,
                    {"left": 0.03},
                ),
                TrajectoryPoint(
                    0.1,
                    {"left": (0.2, -0.3)},
                    chunk.id,
                    {"left": 0.03},
                ),
            ),
            0.1,
        )
        world = WorldState(
            (DetectedObject("red-cube-1", "red cube", 1.0, scene.cube_position, 0.025),),
            {"minimum_clearance_m": 0.08},
            observed_at=1.0,
            coordinate_frame="robot_base",
            up_axis="y",
        )
        driver = MujocoArmDriver(scene, playback_speed=1000.0)

        telemetry = driver.execute_chunk(chunk, trajectory, world)

        assert telemetry.success
        assert driver.steps == ["candidate-transfer"]
        assert scene.cube_position[0] == pytest.approx(platform[0])
        assert scene.cube_position[2] == pytest.approx(platform[2])
        assert scene.cube_position[1] == pytest.approx(platform[1])
    finally:
        scene.close()
