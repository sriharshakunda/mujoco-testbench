"""Controllers package for Agilex Piper arm."""

from src.controllers.ik_controller import (
    DifferentialIKController,
    euler2mat,
    mat2euler,
    mat2quat,
    quat2mat,
)

try:
    from src.controllers.pinocchio_controller import PinocchioIKController
except ImportError:
    PinocchioIKController = None

__all__ = [
    "DifferentialIKController",
    "PinocchioIKController",
    "euler2mat",
    "mat2euler",
    "mat2quat",
    "quat2mat",
]
