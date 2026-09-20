from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _solver_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "solve_co6_extrinsics_from_capture.py"
    spec = importlib.util.spec_from_file_location("co6_extrinsic_solver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pixel_ray_intersects_recovered_board_plane_consistently() -> None:
    solver = _solver_module()
    camera_matrix = np.asarray(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    rotation_camera_from_board = np.eye(3)
    translation_camera_from_board = np.asarray([0.0, 0.0, 1.0])

    center = solver._plane_point_from_pixel(
        (320.0, 240.0),
        camera_matrix,
        distortion,
        rotation_camera_from_board,
        translation_camera_from_board,
    )
    offset = solver._plane_point_from_pixel(
        (400.0, 240.0),
        camera_matrix,
        distortion,
        rotation_camera_from_board,
        translation_camera_from_board,
    )

    np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(offset, [0.1, 0.0, 0.0], atol=1e-12)
