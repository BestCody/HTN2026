"""Solve CO6 camera-to-base calibration from one captured reference frame.

The checkerboard supplies metric scale and a reference plane. Two image
landmarks define the robot frame: the base-yaw center and a point in the
robot's forward direction. Both landmarks are projected onto the checkerboard
plane, so the capture must represent that plane as the robot ground plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.episode_dataset import validate_camera_calibration  # noqa: E402
from moira.robot_config import load_robot_model  # noqa: E402

INNER_CORNERS = (8, 6)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _board_points(square_size_m: float) -> np.ndarray:
    points = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float64)
    points[:, :2] = np.mgrid[0 : INNER_CORNERS[0], 0 : INNER_CORNERS[1]].T.reshape(
        -1, 2
    )
    points *= square_size_m
    return points


def _plane_point_from_pixel(
    pixel: tuple[float, float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation_camera_from_board: np.ndarray,
    translation_camera_from_board: np.ndarray,
) -> np.ndarray:
    """Intersect a distorted image pixel's camera ray with board Z=0.

    Using the recovered PnP pose here is deliberate. A second homography fit
    to noisy corner observations can disagree with that pose outside the
    checkerboard and shift the robot origin by several pixels.
    """
    normalized_pixel = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(2)
    ray_camera = np.asarray(
        [normalized_pixel[0], normalized_pixel[1], 1.0], dtype=np.float64
    )
    rotation_board_from_camera = rotation_camera_from_board.T
    camera_center_board = (
        -rotation_board_from_camera @ translation_camera_from_board
    )
    ray_board = rotation_board_from_camera @ ray_camera
    if abs(float(ray_board[2])) < 1e-9:
        raise RuntimeError("Landmark camera ray is parallel to the reference plane")
    distance = -float(camera_center_board[2]) / float(ray_board[2])
    if not np.isfinite(distance) or distance <= 0.0:
        raise RuntimeError("Landmark camera ray does not meet the plane in front of camera")
    point_board = camera_center_board + distance * ray_board
    point_board[2] = 0.0
    return point_board


def solve(args: argparse.Namespace) -> dict[str, Any]:
    intrinsic = _load(args.intrinsics)
    corner_record = _load(args.corners)
    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read captured image: {args.image}")
    expected_resolution = tuple(int(v) for v in intrinsic["resolution_px"])
    actual_resolution = (image.shape[1], image.shape[0])
    if actual_resolution != expected_resolution:
        raise ValueError(
            f"Capture resolution {actual_resolution} does not match intrinsics "
            f"{expected_resolution}"
        )

    camera_matrix = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        intrinsic["distortion_coefficients"], dtype=np.float64
    )
    corners = np.asarray(corner_record["corners_px"], dtype=np.float64)
    if corners.shape != (INNER_CORNERS[0] * INNER_CORNERS[1], 2):
        raise ValueError("Captured frame must contain exactly 48 ordered corners")
    square_size_m = float(intrinsic["square_size_m"])
    board = _board_points(square_size_m)

    ok, rvec, tvec = cv2.solvePnP(
        board, corners.reshape(-1, 1, 2), camera_matrix, distortion
    )
    if not ok:
        raise RuntimeError("OpenCV could not solve the checkerboard pose")
    rotation_camera_from_board, _ = cv2.Rodrigues(rvec)
    translation_camera_from_board = tvec.reshape(3)

    center_board = _plane_point_from_pixel(
        tuple(args.base_center_px),
        camera_matrix,
        distortion,
        rotation_camera_from_board,
        translation_camera_from_board,
    )
    forward_board = _plane_point_from_pixel(
        tuple(args.base_forward_px),
        camera_matrix,
        distortion,
        rotation_camera_from_board,
        translation_camera_from_board,
    )
    x_axis_board = forward_board - center_board
    x_axis_board[2] = 0.0
    landmark_separation_m = float(np.linalg.norm(x_axis_board))
    if not 0.03 <= landmark_separation_m <= 1.0:
        raise ValueError(
            "Base center/forward landmarks produce an implausible separation: "
            f"{landmark_separation_m:.3f} m"
        )
    x_axis_board /= landmark_separation_m

    camera_center_board = (
        -rotation_camera_from_board.T @ translation_camera_from_board
    )
    y_axis_board = np.asarray(
        [0.0, 0.0, 1.0 if camera_center_board[2] >= 0.0 else -1.0]
    )
    z_axis_board = np.cross(x_axis_board, y_axis_board)
    z_axis_board /= np.linalg.norm(z_axis_board)
    rotation_board_from_base = np.column_stack(
        (x_axis_board, y_axis_board, z_axis_board)
    )
    if not np.allclose(
        rotation_board_from_base.T @ rotation_board_from_base,
        np.eye(3),
        atol=1e-8,
    ) or not np.isclose(np.linalg.det(rotation_board_from_base), 1.0, atol=1e-8):
        raise RuntimeError("Derived robot-base rotation is not right-handed orthonormal")

    camera_from_board = np.eye(4, dtype=np.float64)
    camera_from_board[:3, :3] = rotation_camera_from_board
    camera_from_board[:3, 3] = translation_camera_from_board
    board_from_base = np.eye(4, dtype=np.float64)
    board_from_base[:3, :3] = rotation_board_from_base
    board_from_base[:3, 3] = center_board
    camera_from_base = camera_from_board @ board_from_base
    camera_to_base = np.linalg.inv(camera_from_base)

    projected, _ = cv2.projectPoints(
        board, rvec, tvec, camera_matrix, distortion
    )
    reprojection_rms = float(
        np.sqrt(
            np.mean(
                (projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2
            )
        )
    )
    if not np.isfinite(reprojection_rms) or reprojection_rms > args.max_rms_px:
        raise RuntimeError(
            f"Extrinsic reprojection RMS {reprojection_rms:.3f}px exceeds "
            f"{args.max_rms_px:.3f}px"
        )

    model = load_robot_model(args.robot_model, verify_source=True)
    if model.effective_reach_m is None:
        raise ValueError("Robot model must define kinematics.effective_reach_m")
    reach = float(model.effective_reach_m)
    calibration = {
        "schema_version": 1,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "camera_id": "co6-usb",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "resolution_px": list(actual_resolution),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "camera_to_base_matrix": camera_to_base.tolist(),
        "workspace": {
            "table_height_m": 0.0,
            "bounds_base_frame_m": {
                "x": [0.0, reach],
                "z": [-reach, reach],
            },
        },
        "quality": {
            "intrinsic_reprojection_rms_px": float(
                intrinsic["reprojection_rms_px"]
            ),
            "extrinsic_reprojection_rms_px": reprojection_rms,
            "checkerboard_inner_corners": list(INNER_CORNERS),
            "checkerboard_square_size_m": square_size_m,
            "base_center_px": [float(v) for v in args.base_center_px],
            "base_forward_px": [float(v) for v in args.base_forward_px],
            "base_landmark_separation_on_reference_plane_m": landmark_separation_m,
            "alignment_method": "single_frame_planar_visual_landmarks",
            "reference_plane_assumption": "checkerboard plane equals robot XZ ground plane",
            "physical_alignment_validated": False,
        },
    }
    validate_camera_calibration(calibration, model)
    return calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/camera_workflow/extrinsic_capture.jpg"),
    )
    parser.add_argument(
        "--corners",
        type=Path,
        default=Path("outputs/camera_workflow/extrinsic_capture_corners.json"),
    )
    parser.add_argument(
        "--intrinsics", type=Path, default=Path("outputs/co6_intrinsics.json")
    )
    parser.add_argument(
        "--robot-model",
        type=Path,
        default=Path(
            "robot_models/four_dof_desktop_arm/physical_three_actuator_model.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("config/co6_camera_calibration.json")
    )
    parser.add_argument(
        "--base-center-px", type=float, nargs=2, metavar=("X", "Y"), required=True
    )
    parser.add_argument(
        "--base-forward-px", type=float, nargs=2, metavar=("X", "Y"), required=True
    )
    parser.add_argument("--max-rms-px", type=float, default=2.0)
    args = parser.parse_args()
    calibration = solve(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "extrinsic_rms_px": calibration["quality"][
                    "extrinsic_reprojection_rms_px"
                ],
                "camera_position_base_m": [
                    float(v)
                    for v in np.asarray(calibration["camera_to_base_matrix"])[
                        :3, 3
                    ]
                ],
                "physical_alignment_validated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
