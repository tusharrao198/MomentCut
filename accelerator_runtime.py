"""Dependency-light platform and accelerator policy resolution.

This module deliberately imports no GPU runtime.  Stage adapters perform their
own capability checks only after a route has been selected.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


class AcceleratorConfigurationError(ValueError):
    """An explicit accelerator selection cannot run on this host."""


def is_apple_silicon(*, system=None, machine=None) -> bool:
    system = sys.platform if system is None else system
    machine = platform.machine() if machine is None else machine
    return system == "darwin" and machine.lower() in {"arm64", "aarch64"}


def apple_acceleration_state(value=None) -> str:
    """Return whether Apple-only acceleration adapters may be used.

    ``auto`` enables them only on Apple Silicon. ``enabled`` makes a
    non-Apple host configuration error visible at startup. ``disabled`` keeps
    the portable routes even on an Apple Silicon host.
    """
    requested = (
        os.environ.get("APPLE_SILICON_ACCELERATION", "auto")
        if value is None else value
    )
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "enabled", "disabled"}:
        raise AcceleratorConfigurationError(
            "APPLE_SILICON_ACCELERATION must be auto, enabled, or disabled"
        )
    apple = is_apple_silicon()
    if requested == "enabled" and not apple:
        raise AcceleratorConfigurationError(
            "APPLE_SILICON_ACCELERATION=enabled requires an Apple Silicon Mac"
        )
    if apple and requested != "disabled":
        return "active"
    if apple:
        return "disabled"
    return "not_apple"


def apple_acceleration_active() -> bool:
    """True only when the Mac adapter feature is enabled on Apple Silicon."""
    return apple_acceleration_state() == "active"


def require_apple_acceleration(feature: str) -> None:
    """Reject an explicit Apple-only choice when its feature gate is inactive."""
    state = apple_acceleration_state()
    if state == "active":
        return
    if state == "disabled":
        raise AcceleratorConfigurationError(
            f"{feature} requires Apple acceleration; set "
            "APPLE_SILICON_ACCELERATION=auto or enabled"
        )
    raise AcceleratorConfigurationError(
        f"{feature} requires an Apple Silicon Mac"
    )


def platform_kind() -> str:
    """Return host identity without importing Torch, MLX or CTranslate2."""
    if is_apple_silicon():
        return "apple"
    if shutil.which("nvidia-smi"):
        return "nvidia"
    return "cpu"


def resolve_memory_admission_backend(value=None) -> str:
    """Select the host-wide memory signal used before starting a job."""
    requested = (
        os.environ.get("MEMORY_ADMISSION_BACKEND", "auto")
        if value is None else value
    )
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "nvidia", "macos", "count"}:
        raise AcceleratorConfigurationError(
            "MEMORY_ADMISSION_BACKEND must be auto, nvidia, macos, or count"
        )
    if requested == "macos":
        require_apple_acceleration("MEMORY_ADMISSION_BACKEND=macos")
        return requested
    if requested != "auto":
        return requested
    if apple_acceleration_active():
        return "macos"
    kind = platform_kind()
    if kind == "nvidia":
        return "nvidia"
    return "count"


def nvidia_free_mb():
    """Read free memory on NVIDIA GPU 0, or return None when unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(float(result.stdout.strip().splitlines()[0]))
    except Exception:
        pass
    return None


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
        return "mlx" if apple_acceleration_active() else "faster-whisper"
    if requested == "mlx":
        require_apple_acceleration("WHISPER_ENGINE=mlx")
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
        return "mlx" if apple_acceleration_active() else "onnx"
    if requested == "mlx":
        require_apple_acceleration("PARAKEET_ENGINE=mlx")
    return requested
