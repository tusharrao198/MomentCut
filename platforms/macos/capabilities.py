"""Non-eager native Apple Silicon capability checks.

Run from the OpenShorts checkout:

    ./.venv/bin/python -m platforms.macos.capabilities

The check does not download models, instantiate ASR models, or start a server.
It reports whether the pinned model snapshots and optional runtimes already
exist on this Mac. When opt-in CED audio-event analysis is enabled, it also
runs CED's bounded model/device probe; disabled CED does not touch its model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import accelerator_runtime
import audio_events
from platforms.macos import asr
from platforms.macos import ffmpeg as macos_ffmpeg


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_repository_env():
    """Load checkout configuration without replacing exported settings."""
    from dotenv import load_dotenv

    load_dotenv(REPOSITORY_ROOT / ".env")


def _error(error):
    return f"{type(error).__name__}: {error}"


def _resolved(resolver):
    try:
        return resolver(), ""
    except Exception as error:
        return "unresolved", _error(error)


def _module(name):
    """Test an optional native runtime in a separate Python process.

    MLX extensions can terminate a process during import when Metal is
    unavailable or incompatible.  A capability report must turn that into an
    ordinary unavailable result, not crash the OpenShorts operator command.
    """
    command = [
        sys.executable,
        "-c",
        (
            "import importlib, sys; "
            "module = importlib.import_module(sys.argv[1]); "
            "print(getattr(module, '__version__', ''))"
        ),
        name,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False)
    except Exception as error:
        return {"available": False, "error": _error(error)}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return {
            "available": False,
            "error": detail[-1] if detail else f"exit status {result.returncode}",
        }
    return {"available": True, "version": result.stdout.strip()}


def _snapshot(resolve):
    try:
        repository, revision = resolve()
        path = asr.cached_snapshot_path(repository, revision)
    except Exception as error:
        return {"cached": False, "error": _error(error)}
    return {
        "cached": True,
        "repository": repository,
        "revision": revision,
        "path": path,
    }


def _ffmpeg_capabilities(*, apple_silicon):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    result = {
        "ffmpeg": {"available": bool(ffmpeg), "path": ffmpeg or ""},
        "ffprobe": {"available": bool(ffprobe), "path": ffprobe or ""},
        "videotoolbox": {
            "encoder_listed": False,
            "hwaccel_listed": False,
            "encoder_usable": False,
        },
    }
    if not ffmpeg:
        return result
    try:
        encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30, check=False)
        hwaccels = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=30, check=False)
        result["videotoolbox"]["encoder_listed"] = (
            encoders.returncode == 0 and "h264_videotoolbox" in encoders.stdout)
        result["videotoolbox"]["hwaccel_listed"] = (
            hwaccels.returncode == 0 and "videotoolbox" in hwaccels.stdout)
    except Exception as error:
        result["ffmpeg"]["error"] = _error(error)
        return result
    if apple_silicon:
        result["videotoolbox"]["encoder_usable"] = macos_ffmpeg._probe_encoder()
    return result


def _ced():
    """Validate opt-in CED, including its bounded model/device probe."""
    try:
        settings = audio_events.AudioEventSettings.from_env()
    except Exception as error:
        return {"enabled": True, "ready": False, "error": _error(error)}
    if not settings.enabled:
        return {"enabled": False, "ready": True}
    try:
        from platforms.macos import audio_events as macos_audio_events
        resolved = macos_audio_events.resolve_settings(settings)
    except Exception as error:
        return {"enabled": True, "ready": False, "error": _error(error)}
    return {
        "enabled": True,
        "ready": True,
        "provider": resolved.provider,
        "device": resolved.device,
        "model_sha256": resolved.model_sha256,
        "worker_build": resolved.worker_build,
    }


def _required_models(transcribe_backend):
    # Parakeet may fall back to Whisper, so that mode requires both snapshots.
    return {
        # Whisper is both the default backend and Parakeet's runtime fallback.
        "whisper": True,
        "parakeet": transcribe_backend == "parakeet",
    }


def collect():
    """Return a JSON-serializable report for the native Mac setup."""
    apple_silicon = accelerator_runtime.is_apple_silicon()
    state, state_error = _resolved(accelerator_runtime.apple_acceleration_state)
    transcribe_backend = (
        os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()
        or "whisper"
    )
    required = _required_models(transcribe_backend)
    whisper_engine, whisper_error = _resolved(
        accelerator_runtime.resolve_whisper_engine)
    parakeet_engine, parakeet_error = _resolved(
        accelerator_runtime.resolve_parakeet_engine)
    whisper = {
        "selected_engine": whisper_engine,
        "required": required["whisper"],
        "runtime": _module("mlx_whisper"),
        "model": _snapshot(asr.resolve_whisper_model),
    }
    parakeet = {
        "selected_engine": parakeet_engine,
        "required": required["parakeet"],
        "runtime": _module("parakeet_mlx"),
        "model": _snapshot(asr.resolve_parakeet_model),
    }
    if whisper_error:
        whisper["configuration_error"] = whisper_error
    if parakeet_error:
        parakeet["configuration_error"] = parakeet_error
    fallback = _module("faster_whisper")
    ffmpeg = _ffmpeg_capabilities(apple_silicon=apple_silicon)
    ced = _ced()
    required_ready = all((
        not whisper["required"]
        or (whisper_engine != "mlx" or (
            whisper["runtime"]["available"] and whisper["model"]["cached"])),
        not parakeet["required"]
        or (parakeet_engine != "mlx" or (
            parakeet["runtime"]["available"] and parakeet["model"]["cached"])),
        ffmpeg["ffmpeg"]["available"],
        ffmpeg["ffprobe"]["available"],
        fallback["available"],
        ced["ready"],
    ))
    return {
        "host": {
            "apple_silicon": apple_silicon,
            "apple_acceleration": state,
            "python": sys.version.split()[0],
            **({"configuration_error": state_error} if state_error else {}),
        },
        "transcribe_backend": transcribe_backend,
        "whisper": whisper,
        "parakeet": parakeet,
        "faster_whisper_fallback": fallback,
        "audio_events": ced,
        "ffmpeg": ffmpeg,
        # VideoToolbox is reported but not required: automatic x264 replay is
        # the selected usable fallback when its strict hardware probe fails.
        "ready": bool(
            apple_silicon
            and not state_error
            and state == "active"
            and not whisper_error
            and not parakeet_error
            and required_ready
        ),
    }


def main():
    _load_repository_env()
    report = collect()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
