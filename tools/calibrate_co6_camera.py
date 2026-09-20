"""Interactive intrinsic and robot-base calibration for the Pi-attached CO6 camera."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.brand import DISPLAY_NAME  # noqa: E402
from moira.edge_components import LanCameraSource  # noqa: E402
from moira.episode_dataset import validate_camera_calibration  # noqa: E402
from moira.robot_config import load_robot_model  # noqa: E402

INNER_CORNERS = (8, 6)
DEFAULT_SQUARE_SIZE_M = 0.020
CHECKERBOARD_MEASUREMENT = ROOT / "calibration" / "co6_checkerboard_measurement.json"


def _camera() -> LanCameraSource:
    load_dotenv(ROOT / ".env", override=False)
    url = os.environ.get("MOIRA_CAMERA_URL")
    token = os.environ.get("MOIRA_ROBOT_TOKEN")
    if not url or not token:
        raise RuntimeError("Set MOIRA_CAMERA_URL and MOIRA_ROBOT_TOKEN in .env")
    runtime = _load_json(ROOT / "config" / "pi4_runtime.json")
    rotation = runtime.get("camera", {}).get("rotation_degrees", 0)
    return LanCameraSource(
        url,
        token=token,
        camera_id="co6-usb",
        rotation_degrees=rotation,
    )


def _image(source: LanCameraSource) -> np.ndarray:
    frame = source.capture()
    image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV could not decode the Pi camera JPEG")
    return image


def _corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        INNER_CORNERS,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )
    return corners.astype(np.float32) if found else None


def _board_points(square_size_m: float) -> np.ndarray:
    points = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
    points[:, :2] = np.mgrid[0 : INNER_CORNERS[0], 0 : INNER_CORNERS[1]].T.reshape(
        -1, 2
    )
    points *= square_size_m
    return points


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _configured_square_size_m() -> float:
    """Return the measured size of the physical board used by this robot."""
    if not CHECKERBOARD_MEASUREMENT.is_file():
        return DEFAULT_SQUARE_SIZE_M
    measurement = _load_json(CHECKERBOARD_MEASUREMENT)
    inner_corners = tuple(measurement.get("checkerboard_inner_corners", ()))
    if inner_corners != INNER_CORNERS:
        raise ValueError(
            f"{CHECKERBOARD_MEASUREMENT} does not describe the {INNER_CORNERS} board"
        )
    square_size_m = float(measurement["measured_square_size_m"])
    if not 0.005 <= square_size_m <= 0.1:
        raise ValueError("Configured checkerboard square size is implausible")
    return square_size_m


class _CaptureWindow:
    def __init__(
        self,
        source: LanCameraSource,
        *,
        title: str,
        target_count: int,
        minimum_count: int,
        mark_origin: bool = False,
    ) -> None:
        self.source = source
        self.target_count = target_count
        self.minimum_count = minimum_count
        self.mark_origin = mark_origin
        self.captures: list[tuple[np.ndarray, np.ndarray]] = []
        self.current: tuple[np.ndarray, np.ndarray] | None = None
        self.cancelled = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.status = tk.StringVar(value="Waiting for checkerboard...")
        tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 11)).pack(pady=6)
        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        tk.Button(controls, text="Capture detected view", command=self.capture).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(controls, text="Finish", command=self.finish).pack(side=tk.LEFT, padx=6)
        tk.Button(controls, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=6)
        self._photo: ImageTk.PhotoImage | None = None

    def refresh(self) -> None:
        try:
            image = _image(self.source)
            corners = _corners(image)
            preview = image.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, INNER_CORNERS, corners, True)
                if self.mark_origin:
                    point = tuple(np.rint(corners[0, 0]).astype(int))
                    cv2.circle(preview, point, 9, (0, 0, 255), 3)
                self.current = (image, corners)
                self.status.set(
                    f"Checkerboard detected | captured {len(self.captures)}/{self.target_count}"
                )
            else:
                self.current = None
                self.status.set(
                    f"Checkerboard not detected | captured {len(self.captures)}/{self.target_count}"
                )
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=self._photo)
        except Exception as exc:
            self.current = None
            self.status.set(f"Camera error: {exc}")
        self.root.after(100, self.refresh)

    def capture(self) -> None:
        if self.current is None:
            self.status.set("No checkerboard is currently detected")
            return
        image, corners = self.current
        self.captures.append((image.copy(), corners.copy()))
        if len(self.captures) >= self.target_count:
            self.root.destroy()

    def finish(self) -> None:
        if len(self.captures) < self.minimum_count:
            self.status.set(f"Capture at least {self.minimum_count} valid views")
            return
        self.root.destroy()

    def cancel(self) -> None:
        self.cancelled = True
        self.root.destroy()

    def run(self) -> list[tuple[np.ndarray, np.ndarray]]:
        self.root.after(0, self.refresh)
        self.root.mainloop()
        return [] if self.cancelled else self.captures


class _LiveViewWindow:
    """Continuous, read-only view of the Pi-attached camera."""

    def __init__(self, source: LanCameraSource) -> None:
        self.source = source
        self.closed = False
        self.root = tk.Tk()
        self.root.title(f"{DISPLAY_NAME} | CO6 live camera")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.status = tk.StringVar(value="Connecting to CO6 camera...")
        tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 11)).pack(
            pady=6
        )
        tk.Button(self.root, text="Close", command=self.close, width=14).pack(pady=(0, 8))
        self._photo: ImageTk.PhotoImage | None = None

    def refresh(self) -> None:
        if self.closed:
            return
        try:
            image = _image(self.source)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=self._photo)
            self.status.set(f"LIVE | {image.shape[1]} x {image.shape[0]} | servos untouched")
        except Exception as exc:
            self.status.set(f"Camera error: {exc}")
        self.root.after(100, self.refresh)

    def close(self) -> None:
        self.closed = True
        self.root.destroy()

    def run(self) -> None:
        self.root.after(0, self.refresh)
        self.root.mainloop()


def show_camera(_: argparse.Namespace) -> int:
    _LiveViewWindow(_camera()).run()
    return 0


def capture_intrinsics(args: argparse.Namespace) -> int:
    source = _camera()
    print("Show the printed checkerboard at varied angles and distances.")
    captures = _CaptureWindow(
        source,
        title=f"{DISPLAY_NAME} CO6 intrinsic calibration",
        target_count=args.samples,
        minimum_count=8,
    ).run()
    if not captures:
        return 1
    image_points = [corners for _image_value, corners in captures]
    image_size = (captures[0][0].shape[1], captures[0][0].shape[0])
    object_points = [_board_points(args.square_size_m) for _ in image_points]
    rms, matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if not np.isfinite(rms) or rms > args.max_rms_px:
        raise RuntimeError(
            f"Intrinsic reprojection RMS {rms:.3f}px exceeds {args.max_rms_px:.3f}px"
        )
    result = {
        "schema_version": 1,
        "camera_id": "co6-usb",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "resolution_px": list(image_size),
        "inner_corners": list(INNER_CORNERS),
        "square_size_m": args.square_size_m,
        "views": len(image_points),
        "reprojection_rms_px": float(rms),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rms_px": rms}, indent=2))
    return 0


def capture_extrinsics(args: argparse.Namespace) -> int:
    intrinsic = _load_json(args.intrinsics)
    matrix = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(intrinsic["distortion_coefficients"], dtype=np.float64)
    source = _camera()
    print("Lay the checkerboard flat and aligned with robot-base +X and +Z.")
    print("The red ORIGIN arrow must point to detected corner 0.")
    captures = _CaptureWindow(
        source,
        title=f"{DISPLAY_NAME} CO6 robot-base calibration",
        target_count=1,
        minimum_count=1,
        mark_origin=True,
    ).run()
    if not captures:
        return 1
    image, corners = captures[0]
    if args.reverse_corners:
        corners = corners[::-1].copy()
    square_size_m = float(intrinsic["square_size_m"])
    board = _board_points(square_size_m)
    base_points = np.zeros_like(board)
    base_points[:, 0] = args.origin_x_m + args.x_sign * board[:, 0]
    base_points[:, 1] = args.table_height_m
    base_points[:, 2] = args.origin_z_m + args.z_sign * board[:, 1]
    ok, rotation_vector, translation = cv2.solvePnP(
        base_points,
        corners,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("OpenCV could not solve the camera-to-base transform")
    rotation_base_to_camera, _ = cv2.Rodrigues(rotation_vector)
    rotation_camera_to_base = rotation_base_to_camera.T
    translation_camera_to_base = -rotation_camera_to_base @ translation
    camera_to_base = np.eye(4, dtype=np.float64)
    camera_to_base[:3, :3] = rotation_camera_to_base
    camera_to_base[:3, 3] = translation_camera_to_base.reshape(3)
    projected, _ = cv2.projectPoints(
        base_points, rotation_vector, translation, matrix, distortion
    )
    error = float(np.sqrt(np.mean((projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)))
    if not np.isfinite(error) or error > args.max_rms_px:
        raise RuntimeError(
            f"Extrinsic reprojection RMS {error:.3f}px exceeds {args.max_rms_px:.3f}px"
        )
    model = load_robot_model(args.robot_model, verify_source=True)
    calibration = {
        "schema_version": 1,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "camera_id": "co6-usb",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "resolution_px": [image.shape[1], image.shape[0]],
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "camera_to_base_matrix": camera_to_base.tolist(),
        "workspace": {
            "table_height_m": args.table_height_m,
            "bounds_base_frame_m": {
                "x": [float(base_points[:, 0].min()), float(base_points[:, 0].max())],
                "z": [float(base_points[:, 2].min()), float(base_points[:, 2].max())],
            },
        },
        "quality": {
            "intrinsic_reprojection_rms_px": intrinsic["reprojection_rms_px"],
            "extrinsic_reprojection_rms_px": error,
            "checkerboard_inner_corners": list(INNER_CORNERS),
            "checkerboard_square_size_m": square_size_m,
        },
    }
    validate_camera_calibration(calibration, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rms_px": error}, indent=2))
    return 0


def capture_extrinsic_frame(args: argparse.Namespace) -> int:
    """Capture one checkerboard frame without assuming robot-frame landmarks."""

    source = _camera()
    print("Lay the checkerboard flat in the final robot workspace.")
    captures = _CaptureWindow(
        source,
        title=f"{DISPLAY_NAME} CO6 single-view capture",
        target_count=1,
        minimum_count=1,
        mark_origin=True,
    ).run()
    if not captures:
        return 1
    image, corners = captures[0]
    args.image.parent.mkdir(parents=True, exist_ok=True)
    args.corners.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.image), image):
        raise RuntimeError(f"OpenCV could not save calibration frame: {args.image}")
    corner_record = {
        "schema_version": 1,
        "camera_id": "co6-usb",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "resolution_px": [image.shape[1], image.shape[0]],
        "checkerboard_inner_corners": list(INNER_CORNERS),
        "corners_px": corners.reshape(-1, 2).astype(float).tolist(),
    }
    args.corners.write_text(json.dumps(corner_record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "captured",
                "image": str(args.image.resolve()),
                "corners": str(args.corners.resolve()),
                "detected_corner_count": len(corner_record["corners_px"]),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    intrinsic = commands.add_parser("intrinsics")
    intrinsic.add_argument("--samples", type=int, default=12)
    intrinsic.add_argument(
        "--square-size-m",
        type=float,
        default=_configured_square_size_m(),
        help="measured physical square size (defaults to calibration config)",
    )
    intrinsic.add_argument("--max-rms-px", type=float, default=1.5)
    intrinsic.add_argument(
        "--output", type=Path, default=Path("outputs/co6_intrinsics.json")
    )
    intrinsic.set_defaults(run=capture_intrinsics)
    frame = commands.add_parser("extrinsic-frame")
    frame.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/camera_workflow/extrinsic_capture.jpg"),
    )
    frame.add_argument(
        "--corners",
        type=Path,
        default=Path("outputs/camera_workflow/extrinsic_capture_corners.json"),
    )
    frame.set_defaults(run=capture_extrinsic_frame)
    view = commands.add_parser("view", help="show the continuous read-only camera feed")
    view.set_defaults(run=show_camera)
    extrinsic = commands.add_parser("extrinsics")
    extrinsic.add_argument(
        "--intrinsics", type=Path, default=Path("outputs/co6_intrinsics.json")
    )
    extrinsic.add_argument(
        "--robot-model",
        type=Path,
        default=Path(
            "robot_models/four_dof_desktop_arm/physical_three_actuator_model.json"
        ),
    )
    extrinsic.add_argument(
        "--output", type=Path, default=Path("config/co6_camera_calibration.json")
    )
    extrinsic.add_argument("--origin-x-m", type=float, required=True)
    extrinsic.add_argument("--origin-z-m", type=float, required=True)
    extrinsic.add_argument("--table-height-m", type=float, required=True)
    extrinsic.add_argument("--x-sign", type=int, choices=(-1, 1), default=1)
    extrinsic.add_argument("--z-sign", type=int, choices=(-1, 1), default=1)
    extrinsic.add_argument("--reverse-corners", action="store_true")
    extrinsic.add_argument("--max-rms-px", type=float, default=2.0)
    extrinsic.set_defaults(run=capture_extrinsics)
    args = parser.parse_args()
    if args.command == "intrinsics" and (
        args.samples < 8 or not 0.005 <= args.square_size_m <= 0.1
    ):
        parser.error("--samples must be at least 8 and square size must be 0.005-0.1 m")
    if args.command == "extrinsics" and not all(
        np.isfinite(value)
        for value in (args.origin_x_m, args.origin_z_m, args.table_height_m)
    ):
        parser.error("extrinsic coordinates must be finite")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
