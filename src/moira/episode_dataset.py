"""Atomic, lineage-checked physical-robot episode recording."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .calibration import apply_servo_calibration, write_json_atomic
from .robot_config import RobotModel

SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _load_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value: object, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{name} rows must contain {columns} values")
        result.append([_finite(item, name) for item in row])
    return result


def validate_camera_calibration(
    value: dict[str, Any], model: RobotModel
) -> dict[str, Any]:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Camera calibration schema_version must be {SCHEMA_VERSION}")
    if value.get("robot_model_id") != model.model_id:
        raise ValueError("Camera calibration robot_model_id does not match")
    if value.get("robot_source_sha256") != model.source_sha256:
        raise ValueError("Camera calibration was captured against different robot geometry")
    for name in ("camera_id", "captured_at"):
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Camera calibration {name} must be non-empty")
    resolution = value.get("resolution_px")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in resolution
        )
    ):
        raise ValueError("Camera calibration resolution_px must contain two positive integers")
    camera_matrix = _matrix(value.get("camera_matrix"), 3, 3, "camera_matrix")
    camera_to_base = _matrix(
        value.get("camera_to_base_matrix"), 4, 4, "camera_to_base_matrix"
    )
    distortion = value.get("distortion_coefficients")
    if not isinstance(distortion, list) or not distortion:
        raise ValueError("distortion_coefficients must be a non-empty list")
    distortion_values = [_finite(item, "distortion_coefficients") for item in distortion]
    result = deepcopy(value)
    result["camera_matrix"] = camera_matrix
    result["camera_to_base_matrix"] = camera_to_base
    result["distortion_coefficients"] = distortion_values
    return result


def create_camera_calibration_template(
    robot_model: dict[str, Any], camera_id: str = "co6-usb"
) -> dict[str, Any]:
    model = RobotModel.from_mapping(robot_model)
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera_id must be non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "camera_id": camera_id,
        "captured_at": None,
        "resolution_px": None,
        "camera_matrix": None,
        "distortion_coefficients": None,
        "camera_to_base_matrix": None,
        "workspace": {
            "table_height_m": None,
            "bounds_base_frame_m": None,
        },
    }


def initialize_episode_dataset(
    root: Path,
    robot_model_path: Path,
    servo_calibration_path: Path,
    camera_calibration_path: Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("created_at must be a non-empty timestamp")
    model_value = _load_object(robot_model_path, "robot model")
    model = RobotModel.from_mapping(model_value)
    servo_value = _load_object(servo_calibration_path, "servo calibration")
    apply_servo_calibration(model_value, servo_value)
    camera_value = validate_camera_calibration(
        _load_object(camera_calibration_path, "camera calibration"), model
    )
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Episode dataset directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir()
    (root / ".staging").mkdir()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "robot_model_sha256": _sha256(robot_model_path),
        "servo_calibration_sha256": _sha256(servo_calibration_path),
        "camera_calibration_sha256": _sha256(camera_calibration_path),
        "camera_id": camera_value["camera_id"],
        "created_at": created_at,
        "episode_format": "one-directory-per-atomic-episode",
    }
    write_json_atomic(root / "dataset.json", manifest)
    return manifest


class EpisodeRecorder:
    """Write one episode into staging and publish it with an atomic rename."""

    def __init__(
        self,
        root: Path,
        episode_id: str,
        instruction: str,
        split: str,
        *,
        started_at: str,
    ) -> None:
        if not _SAFE_ID.fullmatch(episode_id):
            raise ValueError("episode_id contains unsupported characters")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if not isinstance(started_at, str) or not started_at.strip():
            raise ValueError("started_at must be a non-empty timestamp")
        self.root = root.resolve()
        self.dataset = _load_object(self.root / "dataset.json", "dataset manifest")
        if self.dataset.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported episode dataset schema")
        self.final_path = self.root / "episodes" / episode_id
        if self.final_path.exists():
            raise FileExistsError(f"Episode already exists: {episode_id}")
        self.stage = self.root / ".staging" / f"{episode_id}.{uuid.uuid4().hex}"
        self.stage.mkdir(parents=False)
        (self.stage / "frames").mkdir()
        self.record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "split": split,
            "instruction": instruction,
            "started_at": started_at,
            "robot_model_id": self.dataset["robot_model_id"],
            "robot_source_sha256": self.dataset["robot_source_sha256"],
            "robot_model_sha256": self.dataset["robot_model_sha256"],
            "servo_calibration_sha256": self.dataset["servo_calibration_sha256"],
            "camera_calibration_sha256": self.dataset["camera_calibration_sha256"],
            "frames": [],
            "transitions": [],
        }
        self._last_frame_ns = -1
        self._last_transition_ns = -1
        self._finished = False

    @staticmethod
    def _timestamp(value: object, name: str, previous: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative monotonic nanosecond integer")
        if value <= previous:
            raise ValueError(f"{name} must increase strictly within an episode")
        return value

    def add_frame(
        self,
        camera_id: str,
        timestamp_ns: int,
        data: bytes,
        *,
        media_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("Episode recorder is already closed")
        if camera_id != self.dataset["camera_id"]:
            raise ValueError("Frame camera_id does not match the dataset calibration")
        if not isinstance(data, bytes) or not data:
            raise ValueError("Frame data must be non-empty bytes")
        timestamp_ns = self._timestamp(timestamp_ns, "frame timestamp_ns", self._last_frame_ns)
        self._last_frame_ns = timestamp_ns
        index = len(self.record["frames"])
        extension = ".jpg" if media_type == "image/jpeg" else ".bin"
        relative = Path("frames") / f"{index:06d}{extension}"
        destination = self.stage / relative
        destination.write_bytes(data)
        entry = {
            "index": index,
            "camera_id": camera_id,
            "timestamp_ns": timestamp_ns,
            "media_type": media_type,
            "path": relative.as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        self.record["frames"].append(entry)
        return entry

    def add_transition(
        self,
        timestamp_ns: int,
        robot_state: dict[str, Any],
        action: dict[str, Any],
        *,
        observation: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError("Episode recorder is already closed")
        timestamp_ns = self._timestamp(
            timestamp_ns, "transition timestamp_ns", self._last_transition_ns
        )
        self._last_transition_ns = timestamp_ns
        for value, name in ((robot_state, "robot_state"), (action, "action")):
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{name} must be a non-empty object")
        if observation is not None and not isinstance(observation, dict):
            raise TypeError("observation must be an object or null")
        entry = {
            "index": len(self.record["transitions"]),
            "timestamp_ns": timestamp_ns,
            "robot_state": deepcopy(robot_state),
            "action": deepcopy(action),
            "observation": deepcopy(observation),
        }
        json.dumps(entry, allow_nan=False)
        self.record["transitions"].append(entry)

    def complete(
        self,
        *,
        completed_at: str,
        success: bool,
        outcome: dict[str, Any],
    ) -> Path:
        if self._finished:
            raise RuntimeError("Episode recorder is already closed")
        if not isinstance(completed_at, str) or not completed_at.strip():
            raise ValueError("completed_at must be a non-empty timestamp")
        if not isinstance(success, bool):
            raise TypeError("success must be boolean")
        if not isinstance(outcome, dict) or not outcome:
            raise ValueError("outcome must be a non-empty object")
        if not self.record["frames"]:
            raise ValueError("An episode must contain at least one camera frame")
        if not self.record["transitions"]:
            raise ValueError("An episode must contain at least one transition")
        self.record.update(
            completed_at=completed_at,
            success=success,
            outcome=deepcopy(outcome),
        )
        json.dumps(self.record, allow_nan=False)
        write_json_atomic(self.stage / "episode.json", self.record)
        os.replace(self.stage, self.final_path)
        self._finished = True
        return self.final_path

    def abort(self) -> None:
        if not self._finished:
            shutil.rmtree(self.stage, ignore_errors=True)
            self._finished = True

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._finished:
            self.abort()
