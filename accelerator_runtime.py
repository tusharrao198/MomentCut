"""Dependency-light platform and accelerator policy resolution.

This module deliberately imports no GPU runtime.  Stage adapters perform their
own capability checks only after a route has been selected.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys


class AcceleratorConfigurationError(ValueError):
    """An explicit accelerator selection cannot run on this host."""


def is_apple_silicon(*, system=None, machine=None) -> bool:
    system = sys.platform if system is None else system
    machine = platform.machine() if machine is None else machine
    return system == "darwin" and machine.lower() in {"arm64", "aarch64"}


def platform_kind() -> str:
    """Return host identity without importing Torch, MLX or CTranslate2."""
    if is_apple_silicon():
        return "apple"
    if shutil.which("nvidia-smi"):
        return "nvidia"
    return "cpu"


def resolve_whisper_engine(value=None) -> str:
    """Resolve the N1 engine while leaving every non-Apple host unchanged."""
    requested = (
        os.environ.get("WHISPER_ENGINE", "auto") if value is None else value
    )
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "faster-whisper", "mlx"}:
        raise AcceleratorConfigurationError(
            "WHISPER_ENGINE must be auto, faster-whisper, or mlx"
        )
    if requested == "auto":
        return "mlx" if is_apple_silicon() else "faster-whisper"
    if requested == "mlx" and not is_apple_silicon():
        raise AcceleratorConfigurationError(
            "WHISPER_ENGINE=mlx requires an Apple Silicon Mac"
        )
    return requested


def resolve_parakeet_engine(value=None) -> str:
    """Resolve N2 without changing the existing ONNX route off Apple."""
    requested = (
        os.environ.get("PARAKEET_ENGINE", "auto") if value is None else value
    )
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "onnx", "mlx"}:
        raise AcceleratorConfigurationError(
            "PARAKEET_ENGINE must be auto, onnx, or mlx"
        )
    if requested == "auto":
        return "mlx" if is_apple_silicon() else "onnx"
    if requested == "mlx" and not is_apple_silicon():
        raise AcceleratorConfigurationError(
            "PARAKEET_ENGINE=mlx requires an Apple Silicon Mac"
        )
    return requested
