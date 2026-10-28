"""macOS-only CED worker adapter.

This module is loaded only for an explicitly enabled CED route.  It never
loads CED in the portable Windows/NVIDIA pipeline.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
from typing import Any, Mapping

import accelerator_runtime
import audio_events


class MacAudioEventsError(audio_events.AudioEventError):
    """A macOS CED worker setup or runtime failure."""


_WORKER_PROTOCOL_VERSION = 1


def _regular_file(value: str, *, executable: bool = False) -> Path:
    candidate = Path(value).expanduser()
    try:
        status = candidate.lstat()
    except OSError as error:
        raise MacAudioEventsError(f"CED path is unavailable: {candidate}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise MacAudioEventsError(f"CED path must be a regular file: {candidate}")
    if executable and not os.access(candidate, os.X_OK):
        raise MacAudioEventsError(f"CED worker is not executable: {candidate}")
    return candidate.resolve(strict=True)


def _read_bounded_json(file_name: Path, maximum_bytes: int) -> dict[str, Any]:
    try:
        status = file_name.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise MacAudioEventsError("CED worker output is not a regular file")
        if status.st_size <= 0 or status.st_size > maximum_bytes:
            raise MacAudioEventsError("CED worker output exceeded its configured size limit")
        with file_name.open("rb") as source:
            value = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MacAudioEventsError(f"CED worker output is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise MacAudioEventsError("CED worker output is not a JSON object")
    return value


def _version(worker: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(worker), "--version", "--json"], check=False,
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MacAudioEventsError(f"Cannot inspect CED worker: {error}") from error
    if result.returncode != 0:
        raise MacAudioEventsError(result.stderr.strip() or "Cannot inspect CED worker")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MacAudioEventsError(
            result.stderr.strip() or f"Cannot inspect CED worker: {error}"
        ) from error
    if not isinstance(value, dict):
        raise MacAudioEventsError("Cannot inspect CED worker")
    if value.get("protocol_version") != _WORKER_PROTOCOL_VERSION:
        raise MacAudioEventsError("CED worker protocol version is unsupported")
    build = value.get("worker_build")
    if not isinstance(build, str) or not build.strip() or len(build) > 4096:
        raise MacAudioEventsError("CED worker did not report a valid build identifier")
    return value


def _probe(settings: audio_events.AudioEventSettings) -> None:
    """Prove the configured worker can load this model on its requested device."""
    command = [
        settings.worker_path, "--probe", "--model", settings.model_path,
        "--requested-device", settings.device,
        "--expected-build-id", settings.worker_build,
        "--parent-pid", str(os.getpid()),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=settings.timeout_seconds, env=_worker_env(settings),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MacAudioEventsError(
            f"Cannot verify CED {settings.device} execution: {error}"
        ) from error
    if result.returncode != 0:
        raise MacAudioEventsError(
            result.stderr.strip() or "CED worker probe failed"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MacAudioEventsError(
            result.stderr.strip() or f"CED worker probe failed: {error}"
        ) from error
    if not isinstance(value, dict):
        raise MacAudioEventsError("CED worker probe failed")
    runtime = value.get("runtime")
    if (
        value.get("protocol_version") != _WORKER_PROTOCOL_VERSION
        or value.get("worker_build") != settings.worker_build
        or value.get("model_sha256") != settings.model_sha256
        or not isinstance(runtime, dict)
        or runtime.get("requested_device") != settings.device
    ):
        raise MacAudioEventsError("CED worker probe returned an unexpected result")
    observed = runtime.get("observed_device")
    if (
        settings.require_accelerator
        and (not isinstance(observed, str) or not observed.upper().startswith("MTL"))
    ):
        raise MacAudioEventsError(
            "CED worker could not initialize the required Metal device"
        )
    if settings.device == "cpu" and (
        not isinstance(observed, str) or observed.lower() != "cpu"
    ):
        raise MacAudioEventsError("CED worker could not initialize the requested CPU device")


def resolve_settings(
    settings: audio_events.AudioEventSettings, *, verify_execution: bool = True,
) -> audio_events.AudioEventSettings:
    """Resolve a CED configuration, optionally probing its execution device."""
    if not settings.enabled:
        return settings
    accelerator_runtime.require_apple_acceleration("AUDIO_EVENTS_MODE=ced")
    if settings.device == "metal" and not settings.require_accelerator:
        raise MacAudioEventsError("Metal CED jobs must require the accelerator")
    model = _regular_file(settings.model_path)
    worker = _regular_file(settings.worker_path, executable=True)
    version = _version(worker)
    if settings.device == "metal" and version.get("metal_compiled") is not True:
        raise MacAudioEventsError("CED worker was not built with Metal support")
    resolved = replace(
        settings,
        model_path=str(model),
        worker_path=str(worker),
        model_sha256=audio_events._sha256_file(str(model)),
        worker_build=version["worker_build"],
    )
    if verify_execution:
        _probe(resolved)
    return resolved


def settings_for_execution(durable: Mapping[str, Any]) -> audio_events.AudioEventSettings:
    """Reconstruct only the sealed structure, without requiring a live worker."""
    settings = audio_events.AudioEventSettings.from_durable_config(durable)
    if settings.enabled:
        accelerator_runtime.require_apple_acceleration("AUDIO_EVENTS_MODE=ced")
    return settings


def resolve_worker_for_new_run(settings: audio_events.AudioEventSettings) -> None:
    """Recheck mutable local toolchain immediately before Popen."""
    if not settings.enabled:
        return
    model = _regular_file(settings.model_path)
    worker = _regular_file(settings.worker_path, executable=True)
    if audio_events._sha256_file(str(model)) != settings.model_sha256:
        raise MacAudioEventsError("CED model no longer matches the accepted job")
    version = _version(worker)
    if version.get("worker_build") != settings.worker_build:
        raise MacAudioEventsError("CED worker no longer matches the accepted job")
    if settings.device == "metal" and version.get("metal_compiled") is not True:
        raise MacAudioEventsError("CED worker no longer has Metal support")


def _worker_env(settings: audio_events.AudioEventSettings) -> dict[str, str]:
    names = ("PATH", "HOME", "TMPDIR", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")
    result = {name: os.environ[name] for name in names if os.environ.get(name)}
    result["CED_DEVICE"] = "MTL0" if settings.device == "metal" else "cpu"
    return result


def _stop_worker(process: subprocess.Popen) -> None:
    """Terminate a CED process group, escalating only when needed."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def run(
    settings: audio_events.AudioEventSettings,
    *,
    source_file: str,
    source_duration_seconds: float,
    audio_stream: Mapping[str, Any],
    project_dir: str,
    ffmpeg_path: str = "ffmpeg",
) -> dict[str, Any]:
    """Extract selected PCM then execute one bounded native CED worker."""
    if not settings.enabled:
        raise MacAudioEventsError("Cannot run a disabled audio-event stage")
    if not audio_events.runtime_enabled():
        raise MacAudioEventsError("Audio-event processing is disabled by AUDIO_EVENTS_RUNTIME_ENABLED")
    timeline_duration = audio_events.source_audio_duration(
        source_duration_seconds, audio_stream.get("start_seconds", 0.0),
    )
    resolve_worker_for_new_run(settings)
    project = Path(project_dir).resolve(strict=True)
    if not project.is_dir():
        raise MacAudioEventsError("Job project directory is unavailable")
    with tempfile.TemporaryDirectory(prefix="audio-events.", dir=project) as temp_dir:
        temporary = Path(temp_dir)
        wav_file = temporary / "selected-stream.wav"
        raw_output = temporary / "ced-result.json"
        decoded_duration = audio_events.extract_pcm_wav(
            ffmpeg_path, source_file, audio_stream, str(wav_file),
            settings.expected_sample_rate_hz, settings.timeout_seconds,
        )
        audio_events.window_count(
            min(decoded_duration, timeline_duration), settings,
        )
        command = [
            settings.worker_path, "--model", settings.model_path,
            "--input", str(wav_file), "--output", str(raw_output),
            "--window-seconds", str(settings.window_seconds),
            "--hop-seconds", str(settings.hop_seconds),
            "--max-windows", str(settings.max_windows),
            "--max-output-bytes", str(settings.max_raw_bytes),
            "--expected-sample-rate-hz", str(settings.expected_sample_rate_hz),
            "--audio-stream-start-seconds", str(audio_stream["start_seconds"]),
            "--source-duration-seconds", str(source_duration_seconds),
            "--requested-device", settings.device,
            "--expected-build-id", settings.worker_build,
            "--parent-pid", str(os.getpid()),
        ]
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=_worker_env(settings),
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=settings.timeout_seconds)
            except subprocess.TimeoutExpired:
                _stop_worker(process)
                raise MacAudioEventsError("CED worker timed out")
            except KeyboardInterrupt:
                _stop_worker(process)
                raise
        except OSError as error:
            raise MacAudioEventsError(f"Cannot start CED worker: {error}") from error
        if process.returncode != 0:
            detail = (stderr or stdout).strip()
            raise MacAudioEventsError(detail or f"CED worker failed with status {process.returncode}")
        return _read_bounded_json(raw_output, settings.max_raw_bytes)
