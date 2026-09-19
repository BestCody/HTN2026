"""Fusion entry point for the repository's robot geometry exporter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _implementation() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "fusion_export_robot.py"
    spec = importlib.util.spec_from_file_location("moira_fusion_export_robot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the MoIRA Fusion exporter at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(context: Any) -> None:
    _implementation().run(context)


def stop(context: Any) -> None:
    _implementation().stop(context)

