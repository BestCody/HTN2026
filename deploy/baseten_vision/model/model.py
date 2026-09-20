"""Calibrated Grounding DINO endpoint matching MoIRA's perception contract."""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
from collections import defaultdict
from typing import Any

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_LABELS = 32


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "object"


def _project_to_plane(
    calibration: dict[str, Any],
    pixel_x: float,
    pixel_y: float,
    up_axis: str,
    plane_height_m: float,
) -> list[float]:
    import cv2
    import numpy as np

    def matrix(name: str, rows: int, columns: int) -> Any:
        value = calibration.get(name)
        if (
            not isinstance(value, list)
            or len(value) != rows
            or any(not isinstance(row, list) or len(row) != columns for row in value)
        ):
            raise ValueError(f"{name} must be a {rows}x{columns} numeric array")
        return np.asarray(
            [[_finite_number(item, name) for item in row] for row in value],
            dtype=np.float64,
        )

    camera_matrix = matrix("camera_matrix", 3, 3)
    camera_to_base = matrix("camera_to_base_matrix", 4, 4)
    distortion = calibration.get("distortion_coefficients")
    if not isinstance(distortion, list) or len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError("distortion_coefficients must use an OpenCV coefficient count")
    distortion_array = np.asarray(
        [_finite_number(value, "distortion coefficient") for value in distortion],
        dtype=np.float64,
    )
    if not np.allclose(camera_to_base[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera_to_base_matrix must be a homogeneous transform")
    rotation = camera_to_base[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
        raise ValueError("camera_to_base_matrix rotation must be orthonormal")
    normalized = cv2.undistortPoints(
        np.asarray([[[pixel_x, pixel_y]]], dtype=np.float64),
        camera_matrix,
        distortion_array,
    )[0, 0]
    ray_camera = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    ray_base = rotation @ ray_camera
    origin_base = camera_to_base[:3, 3]
    up_index = {"x": 0, "y": 1, "z": 2}[up_axis]
    if math.isclose(float(ray_base[up_index]), 0.0, abs_tol=1e-9):
        raise ValueError("camera ray is parallel to the support plane")
    distance = (plane_height_m - origin_base[up_index]) / ray_base[up_index]
    if not math.isfinite(float(distance)) or distance <= 0:
        raise ValueError("detected object projects behind the calibrated camera")
    return [float(value) for value in origin_base + distance * ray_base]


class Model:
    def __init__(self, **kwargs: Any) -> None:
        config = kwargs["config"]
        self._model_path = config["model_metadata"]["model_id"]
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            local_files_only=True,
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_path,
            local_files_only=True,
        ).to("cuda").eval()

    @staticmethod
    def _decode_frame(frame: dict[str, Any]) -> Any:
        from PIL import Image, UnidentifiedImageError

        data = frame.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("base64"), str):
            raise ValueError("frame.data.base64 must contain a base64-encoded image")
        try:
            raw = base64.b64decode(data["base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("frame.data.base64 is not valid base64") from exc
        if not raw:
            raise ValueError("frame image is empty")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("frame image exceeds the 8 MiB limit")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("frame data is not a supported image") from exc
        return image.convert("RGB")

    @staticmethod
    def _perception_config(workspace: dict[str, Any]) -> dict[str, Any]:
        value = workspace.get("perception")
        if not isinstance(value, dict):
            raise ValueError("workspace.perception configuration is required")
        labels = value.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or len(labels) > MAX_LABELS
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len({label.strip().lower() for label in labels}) != len(labels)
        ):
            raise ValueError("workspace.perception.labels needs 1-32 unique labels")
        value = dict(value)
        value["labels"] = [label.strip().lower() for label in labels]
        return value

    @staticmethod
    def _camera_calibration(config: dict[str, Any], camera_id: str) -> dict[str, Any]:
        calibrations = config.get("camera_calibrations")
        if not isinstance(calibrations, dict) or not isinstance(
            calibrations.get(camera_id), dict
        ):
            raise ValueError(f"calibration is missing for camera {camera_id}")
        calibration = calibrations[camera_id]
        return calibration

    @staticmethod
    def _match_label(predicted: str, requested: list[str]) -> str | None:
        value = predicted.strip().lower().strip(" .")
        exact = [label for label in requested if value == label]
        if exact:
            return exact[0]
        matches = [label for label in requested if label in value or value in label]
        return max(matches, key=len) if matches else None

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Vision model is not loaded")
        frames = request.get("frames")
        workspace = request.get("workspace")
        if not isinstance(frames, list) or not frames:
            raise ValueError("frames must contain at least one camera frame")
        if not isinstance(workspace, dict):
            raise ValueError("workspace must be an object")
        config = self._perception_config(workspace)
        threshold = _finite_number(config.get("confidence_threshold", 0.32), "threshold")
        text_threshold = _finite_number(
            config.get("text_threshold", 0.25), "text threshold"
        )
        if not 0 < threshold <= 1 or not 0 < text_threshold <= 1:
            raise ValueError("perception thresholds must be in (0, 1]")
        dimensions = config.get("object_dimensions_m")
        if not isinstance(dimensions, dict):
            raise ValueError("workspace.perception.object_dimensions_m is required")
        table_height = _finite_number(
            workspace.get("table_height_m"), "workspace.table_height_m"
        )
        up_axis = workspace.get("up_axis")
        coordinate_frame = workspace.get("coordinate_frame")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise ValueError("workspace.coordinate_frame must be a non-empty string")
        if up_axis not in ("x", "y", "z"):
            raise ValueError("workspace.up_axis must be x, y, or z")
        hazard_labels = {
            str(value).strip().lower() for value in config.get("hazard_labels", [])
        }

        detections: list[dict[str, Any]] = []
        last_observed = 0.0
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("camera_id"), str):
                raise ValueError("every frame needs a camera_id")
            camera_id = frame["camera_id"]
            calibration = self._camera_calibration(config, camera_id)
            image = self._decode_frame(frame)
            resolution = calibration.get("resolution_px")
            if resolution != [image.width, image.height]:
                raise ValueError(
                    f"camera {camera_id} frame resolution does not match its calibration"
                )
            inputs = self._processor(
                images=image,
                text=[config["labels"]],
                return_tensors="pt",
            ).to("cuda")
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            for box, score, predicted in zip(
                result["boxes"], result["scores"], result["labels"], strict=True
            ):
                label = self._match_label(str(predicted), config["labels"])
                if label is None:
                    continue
                size = dimensions.get(label)
                if (
                    not isinstance(size, list)
                    or len(size) != 3
                    or any(_finite_number(value, f"{label} dimension") <= 0 for value in size)
                ):
                    raise ValueError(f"positive 3D dimensions are required for label {label}")
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                contact_position = _project_to_plane(
                    calibration,
                    (x1 + x2) / 2,
                    y2,
                    up_axis,
                    table_height,
                )
                height_index = {"x": 0, "y": 1, "z": 2}[up_axis]
                contact_position[height_index] += float(size[height_index]) / 2
                detections.append(
                    {
                        "label": label,
                        "confidence": float(score.item()),
                        "position_m": contact_position,
                        "attributes": {
                            "bbox_px": [x1, y1, x2, y2],
                            "dimensions_m": [float(item) for item in size],
                            "camera_id": camera_id,
                        },
                    }
                )
            last_observed = max(
                last_observed,
                _finite_number(frame.get("captured_at"), "frame.captured_at"),
            )

        detections.sort(
            key=lambda item: (
                item["label"],
                item["position_m"][0],
                item["position_m"][1],
            )
        )
        counts: defaultdict[str, int] = defaultdict(int)
        objects = []
        for item in detections:
            counts[item["label"]] += 1
            item["id"] = f"{_slug(item['label'])}-{counts[item['label']]}"
            objects.append(item)
        return {
            "objects": objects,
            "workspace": workspace,
            "hazards": [item["id"] for item in objects if item["label"] in hazard_labels],
            "observed_at": last_observed,
            "geometry": {item["id"]: item["attributes"] for item in objects},
            "coordinate_frame": coordinate_frame,
            "up_axis": up_axis,
        }
