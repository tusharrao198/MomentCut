"""Validated worker-to-API failure records.

The video worker is a subprocess, so exception types cannot reach ``app.py``.
This module carries only a small allow-listed terminal record between them.
It imports no model, accelerator, web, or application code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from stat import S_ISREG
import sys
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 1
RECORD_NAME = ".worker-failure.json"
MAX_RECORD_BYTES = 4096
_ENV_PREFIX = "OPENSHORTS_WORKER_FAILURE_"

CODES = frozenset({
    "resource_exhausted", "backend_busy", "backend_unavailable",
    "unsupported_operation", "service_unavailable", "invalid_input",
    "incomplete_output", "unknown",
})
STAGES = frozenset({"input", "whisper", "parakeet", "clip_selection", "render", "unknown"})
BACKENDS = frozenset({
    "cpu", "cuda", "faster-whisper", "mlx", "mps", "onnx", "videotoolbox", "ffmpeg",
    "ollama", "gemini", "unknown",
})
AUTO_RETRY_CODES = frozenset({
    "resource_exhausted", "backend_busy", "service_unavailable", "incomplete_output",
})


@dataclass(frozen=True)
class FailureIdentity:
    """Values the API expects from exactly one worker execution."""

    job_id: str
    configuration_revision: int
    execution_generation: int

    def __post_init__(self):
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        for name in ("configuration_revision", "execution_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class WorkerFailure(RuntimeError):
    """An allow-listed terminal outcome, never a free-form exception channel."""

    def __init__(self, code: str, stage: str = "unknown", backend: str = "unknown",
                 attempted_backends=()):
        self.code = _allowed(code, CODES, "code")
        self.stage = _allowed(stage, STAGES, "stage")
        self.backend = _allowed(backend, BACKENDS, "backend")
        self.attempted_backends = _backends(attempted_backends)
        super().__init__(f"{self.stage}: {self.code}")


def records_enabled(value=None, *, apple_acceleration_active=None) -> bool:
    """Whether the structured contract is on for this host.

    ``auto`` (the default) keeps existing Windows/NVIDIA retry behavior and
    follows the additive Apple feature gate. ``enabled`` is an explicit
    cross-platform diagnostic opt-in; ``disabled`` restores legacy matching.
    """
    raw = os.environ.get("WORKER_FAILURE_RECORDS", "auto") if value is None else value
    raw = str(raw or "auto").strip().lower()
    if raw not in {"auto", "enabled", "disabled"}:
        raise ValueError("WORKER_FAILURE_RECORDS must be auto, enabled, or disabled")
    if raw == "enabled":
        return True
    if raw == "disabled":
        return False
    if apple_acceleration_active is None:
        from accelerator_runtime import apple_acceleration_active
        apple_acceleration_active = apple_acceleration_active()
    return bool(apple_acceleration_active)


def failure_path(output_dir) -> Path:
    return Path(output_dir) / RECORD_NAME


def clear_record(output_dir) -> None:
    try:
        failure_path(output_dir).unlink()
    except OSError:
        pass


def write_record(output_dir, identity: FailureIdentity, failure: WorkerFailure) -> dict:
    """Atomically write the one validated terminal record for this execution."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "job_id": identity.job_id,
        "configuration_revision": identity.configuration_revision,
        "execution_generation": identity.execution_generation,
        "code": failure.code,
        "backend": failure.backend,
        "stage": failure.stage,
        "fallback_exhausted": True,
        "attempted_backends": list(failure.attempted_backends),
    }
    _validate_record(record, identity)
    path = failure_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".worker-failure-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return record


def read_record(output_dir, identity: FailureIdentity):
    """Return a matching validated record, otherwise ``None`` without retry risk."""
    path = failure_path(output_dir)
    try:
        stat = path.lstat()
        if not S_ISREG(stat.st_mode) or stat.st_size <= 0 or stat.st_size > MAX_RECORD_BYTES:
            return None
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        return _validate_record(record, identity)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def should_auto_retry(record: Mapping[str, Any]) -> bool:
    """API retry policy for a validated terminal record."""
    return str(record.get("code") or "") in AUTO_RETRY_CODES


def reporter_from_environment(output_dir, environ=None):
    """Create a reporter only when app.py provided a complete identity."""
    source = os.environ if environ is None else environ
    try:
        identity = FailureIdentity(
            job_id=str(source[f"{_ENV_PREFIX}JOB_ID"]),
            configuration_revision=int(source[f"{_ENV_PREFIX}CONFIGURATION_REVISION"]),
            execution_generation=int(source[f"{_ENV_PREFIX}EXECUTION_GENERATION"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return FailureReporter(output_dir, identity)


class FailureReporter:
    """One process-local reporter. The API validates its file independently."""

    def __init__(self, output_dir, identity: FailureIdentity):
        self.output_dir = output_dir
        self.identity = identity
        self._failure = None
        self._context = ("unknown", "unknown", ())

    def set_context(self, stage: str, backend: str = "unknown", attempted_backends=()):
        self._context = (
            _allowed(stage, STAGES, "stage"),
            _allowed(backend, BACKENDS, "backend"),
            _backends(attempted_backends),
        )

    def record(self, error: BaseException):
        if self._failure is not None:
            return self._failure
        stage, backend, attempted = self._context
        if isinstance(error, WorkerFailure):
            failure = error
        elif _is_accelerator_configuration_error(error):
            failure = WorkerFailure("backend_unavailable", stage, backend, attempted)
        else:
            failure = WorkerFailure("unknown", stage, backend, attempted)
        self._failure = write_record(self.output_dir, self.identity, failure)
        return self._failure

    def install_uncaught_exception_hook(self):
        previous = sys.excepthook

        def report(exc_type, error, traceback):
            if not issubclass(exc_type, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                try:
                    self.record(error)
                except Exception:
                    pass
            previous(exc_type, error, traceback)

        sys.excepthook = report
        return self


def _is_accelerator_configuration_error(error: BaseException) -> bool:
    try:
        from accelerator_runtime import AcceleratorConfigurationError
        return isinstance(error, AcceleratorConfigurationError)
    except Exception:
        return False


def _allowed(value, allowed, name):
    value = str(value or "").strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} is not supported")
    return value


def _backends(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("attempted_backends must be a list or tuple")
    if len(values) > len(BACKENDS):
        raise ValueError("attempted_backends is too long")
    normalized = tuple(_allowed(value, BACKENDS, "attempted backend") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("attempted_backends must not repeat a backend")
    return normalized


def _validate_record(record, identity: FailureIdentity):
    required = {
        "schema_version", "job_id", "configuration_revision", "execution_generation",
        "code", "backend", "stage", "fallback_exhausted", "attempted_backends",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("worker failure record has an invalid schema")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("worker failure record schema version is unsupported")
    candidate = FailureIdentity(
        job_id=record["job_id"],
        configuration_revision=record["configuration_revision"],
        execution_generation=record["execution_generation"],
    )
    if candidate != identity:
        raise ValueError("worker failure record identity does not match this execution")
    if record["fallback_exhausted"] is not True:
        raise ValueError("worker failure record must represent exhausted fallback")
    failure = WorkerFailure(
        record["code"], record["stage"], record["backend"], record["attempted_backends"])
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": identity.job_id,
        "configuration_revision": identity.configuration_revision,
        "execution_generation": identity.execution_generation,
        "code": failure.code,
        "backend": failure.backend,
        "stage": failure.stage,
        "fallback_exhausted": True,
        "attempted_backends": list(failure.attempted_backends),
    }
