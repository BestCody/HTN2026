"""Factories that bind local physical authority to one validated robot model."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .components import ModelComponent
from .edge_components import LoadFeedbackComponent
from .physical import ArmDriver, BimanualControlComponent
from .robot_config import RobotModel
from .specialists import (
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    PlanarBimanualIK,
    TelemetryFailureClassifier,
)


def robot_bound_local_factories(
    model: RobotModel,
    left: ArmDriver,
    right: ArmDriver | None = None,
) -> Mapping[str, Callable[[], ModelComponent]]:
    """Build manifest factories whose physical limits all come from ``model``."""

    model.require_motion_ready()
    return {
        "local-bimanual-ik": lambda: PlanarBimanualIK.from_robot_model(model),
        "local-trajectory-planner": lambda: BoundedTrajectoryPlanner.from_robot_model(model),
        "local-collision-checker": lambda: ConservativeCollisionChecker.from_robot_model(model),
        "local-safety-risk": lambda: HardSafetyRiskModel.from_robot_model(model),
        "dual-arm-hardware": lambda: BimanualControlComponent.from_installed_arms(
            model,
            left=left,
            right=right,
        ),
        "local-failure-classifier": lambda: TelemetryFailureClassifier.from_robot_model(model),
        "local-load-feedback": lambda: LoadFeedbackComponent.from_robot_model(model),
    }
