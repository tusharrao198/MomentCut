"""Provider-neutral audio-event configuration and artifact handling.

This module intentionally has no accelerator, model, or platform imports.  It
is used by job-state/checkpoint code as well as the macOS CED adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
import wave


class AudioEventError(RuntimeError):
    """A user-safe failure in audio-event preparation or validation."""


_AUDIO_EVENT_SCHEMA_VERSION = 1
_AUDIO_EVENT_MAX_LABELS = 1024
_AUDIO_EVENT_MAX_WINDOWS = 10_000
_AUDIO_EVENT_MAX_EVENTS = 10_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE = {
    "id": "gameplay-v1",
    "version": 1,
    "sample_rate_hz": 16_000,
    "labels": {
        "Gunshot, gunfire": 0.10,
        "Machine gun": 0.12,
        "Fusillade": 0.12,
        "Artillery fire": 0.12,
        "Explosion": 0.10,
        "Boom": 0.16,
    },
    "merge_gap_seconds": 2.0,
    "candidate_padding_seconds": 5.0,
    "max_evidence_per_candidate": 5,
}

# These are semantic inputs and must be copied into a sealed job config.  The
# runtime kill switch deliberately is excluded so it can stop an in-flight
# attempt without changing its compatibility identity.
AUDIO_EVENT_ENV_NAMES = (
    "AUDIO_EVENTS_MODE",
    "AUDIO_EVENTS_PROVIDER",
    "AUDIO_EVENTS_DEVICE",
    "AUDIO_EVENTS_REQUIRE_ACCELERATOR",
    "AUDIO_EVENTS_ALLOW_CPU_FALLBACK",
    "AUDIO_EVENTS_MODEL_PATH",
    "AUDIO_EVENTS_CED_WORKER_PATH",
    "AUDIO_EVENTS_WINDOW_SECONDS",
    "AUDIO_EVENTS_HOP_SECONDS",
    "AUDIO_EVENTS_PROFILE",
    "AUDIO_EVENTS_MAX_WINDOWS",
    "AUDIO_EVENTS_MAX_RAW_BYTES",
    "AUDIO_EVENTS_MAX_CANDIDATE_WINDOWS",
    "AUDIO_EVENTS_TIMEOUT_SECONDS",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _profile_digest() -> str:
    return hashlib.sha256(_canonical_bytes(_PROFILE)).hexdigest()


def _as_bool(name: str, value: Any) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise AudioEventError(f"{name} must be exactly true or false")


def _positive_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AudioEventError(f"{name} must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise AudioEventError(f"{name} must be a finite positive number")
    return parsed


def _positive_int(name: str, value: Any, *, maximum: int | None = None) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise AudioEventError(f"{name} must be a positive integer") from error
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise AudioEventError(f"{name} must be a positive integer")
    if maximum is not None and parsed > maximum:
        raise AudioEventError(f"{name} cannot exceed {maximum}")
    return parsed


def _sample_count(name: str, seconds: float, sample_rate_hz: int) -> int:
    """Return an exact PCM sample count for a sealed duration."""
    samples = seconds * sample_rate_hz
    rounded = round(samples)
    if rounded <= 0 or not math.isclose(samples, rounded, abs_tol=1e-6):
        raise AudioEventError(
            f"{name} must be an exact multiple of 1/{sample_rate_hz} seconds"
        )
    return rounded


def _sha256_file(file_name: str) -> str:
    digest = hashlib.sha256()
    with open(file_name, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strict_json_number(value: Any) -> float | None:
    """Return only a finite JSON number, never a boolean or numeric string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _finite_number(value)


def _bounded_string(value: Any, *, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result if result and len(result) <= limit else None


@dataclass(frozen=True)
class AudioEventSettings:
    """Sealable settings without filesystem or Apple runtime access."""

    mode: str
    provider: str = ""
    device: str = ""
    require_accelerator: bool = False
    allow_cpu_fallback: bool = False
    model_path: str = ""
    worker_path: str = ""
    window_seconds: float = 10.0
    hop_seconds: float = 2.0
    profile_id: str = "gameplay-v1"
    max_windows: int = 1000
    max_raw_bytes: int = 33_554_432
    max_candidate_windows: int = 24
    timeout_seconds: int = 900
    profile_version: int = 1
    profile_sha256: str = ""
    expected_sample_rate_hz: int = 16_000
    model_sha256: str = ""
    worker_build: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode == "ced"

    @classmethod
    def from_env(cls, environ: Mapping[str, Any] | None = None) -> "AudioEventSettings":
        env = os.environ if environ is None else environ
        mode = str(env.get("AUDIO_EVENTS_MODE", "disabled") or "disabled").strip().lower()
        if mode not in {"disabled", "ced"}:
            raise AudioEventError("AUDIO_EVENTS_MODE must be disabled or ced")
        if mode == "disabled":
            return cls(mode="disabled", profile_sha256=_profile_digest())
        provider = str(env.get("AUDIO_EVENTS_PROVIDER", "ced") or "").strip().lower()
        device = str(env.get("AUDIO_EVENTS_DEVICE", "metal") or "").strip().lower()
        profile_id = str(env.get("AUDIO_EVENTS_PROFILE", "gameplay-v1") or "").strip()
        if provider != "ced":
            raise AudioEventError("AUDIO_EVENTS_PROVIDER must be ced")
        if device not in {"metal", "cpu"}:
            raise AudioEventError("AUDIO_EVENTS_DEVICE must be metal or cpu")
        if profile_id != _PROFILE["id"]:
            raise AudioEventError(f"AUDIO_EVENTS_PROFILE must be {_PROFILE['id']}")
        require_accelerator = _as_bool(
            "AUDIO_EVENTS_REQUIRE_ACCELERATOR",
            env.get("AUDIO_EVENTS_REQUIRE_ACCELERATOR", "true"),
        )
        allow_cpu_fallback = _as_bool(
            "AUDIO_EVENTS_ALLOW_CPU_FALLBACK",
            env.get("AUDIO_EVENTS_ALLOW_CPU_FALLBACK", "false"),
        )
        if require_accelerator == allow_cpu_fallback:
            raise AudioEventError(
                "AUDIO_EVENTS_REQUIRE_ACCELERATOR and "
                "AUDIO_EVENTS_ALLOW_CPU_FALLBACK must have opposite values"
            )
        if device == "metal" and not require_accelerator:
            raise AudioEventError("AUDIO_EVENTS_DEVICE=metal requires accelerator")
        if device == "cpu" and require_accelerator:
            raise AudioEventError("AUDIO_EVENTS_DEVICE=cpu requires CPU fallback")
        model_path = str(env.get("AUDIO_EVENTS_MODEL_PATH", "") or "").strip()
        worker_path = str(env.get("AUDIO_EVENTS_CED_WORKER_PATH", "") or "").strip()
        if not model_path or not worker_path:
            raise AudioEventError(
                "AUDIO_EVENTS_MODEL_PATH and AUDIO_EVENTS_CED_WORKER_PATH are required"
            )
        window_seconds = _positive_float(
            "AUDIO_EVENTS_WINDOW_SECONDS", env.get("AUDIO_EVENTS_WINDOW_SECONDS", "10")
        )
        hop_seconds = _positive_float(
            "AUDIO_EVENTS_HOP_SECONDS", env.get("AUDIO_EVENTS_HOP_SECONDS", "2")
        )
        if hop_seconds > window_seconds:
            raise AudioEventError("AUDIO_EVENTS_HOP_SECONDS cannot exceed window seconds")
        _sample_count(
            "AUDIO_EVENTS_WINDOW_SECONDS",
            window_seconds,
            _PROFILE["sample_rate_hz"],
        )
        _sample_count(
            "AUDIO_EVENTS_HOP_SECONDS",
            hop_seconds,
            _PROFILE["sample_rate_hz"],
        )
        return cls(
            mode=mode,
            provider=provider,
            device=device,
            require_accelerator=require_accelerator,
            allow_cpu_fallback=allow_cpu_fallback,
            model_path=model_path,
            worker_path=worker_path,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            profile_id=profile_id,
            max_windows=_positive_int(
                "AUDIO_EVENTS_MAX_WINDOWS",
                env.get("AUDIO_EVENTS_MAX_WINDOWS", "1000"),
                maximum=_AUDIO_EVENT_MAX_WINDOWS,
            ),
            max_raw_bytes=_positive_int(
                "AUDIO_EVENTS_MAX_RAW_BYTES", env.get("AUDIO_EVENTS_MAX_RAW_BYTES", "33554432")
            ),
            max_candidate_windows=_positive_int(
                "AUDIO_EVENTS_MAX_CANDIDATE_WINDOWS",
                env.get("AUDIO_EVENTS_MAX_CANDIDATE_WINDOWS", "24"),
                maximum=_AUDIO_EVENT_MAX_EVENTS,
            ),
            timeout_seconds=_positive_int(
                "AUDIO_EVENTS_TIMEOUT_SECONDS", env.get("AUDIO_EVENTS_TIMEOUT_SECONDS", "900")
            ),
            profile_version=_PROFILE["version"],
            profile_sha256=_profile_digest(),
            expected_sample_rate_hz=_PROFILE["sample_rate_hz"],
        )

    @classmethod
    def from_durable_config(cls, value: Mapping[str, Any]) -> "AudioEventSettings":
        if not valid_durable_config(value):
            raise AudioEventError("The saved audio-events configuration is invalid")
        if value.get("enabled") is False:
            return cls(mode="disabled", profile_sha256=_profile_digest())
        return cls(
            mode="ced",
            provider=value["provider"],
            device=value["device"],
            require_accelerator=value["require_accelerator"],
            allow_cpu_fallback=value["allow_cpu_fallback"],
            model_path=value["model_path"],
            worker_path=value["worker_path"],
            window_seconds=float(value["window_seconds"]),
            hop_seconds=float(value["hop_seconds"]),
            profile_id=value["profile_id"],
            max_windows=value["max_windows"],
            max_raw_bytes=value["max_raw_bytes"],
            max_candidate_windows=value["max_candidate_windows"],
            timeout_seconds=value["timeout_seconds"],
            profile_version=value["profile_version"],
            profile_sha256=value["profile_sha256"],
            expected_sample_rate_hz=value["expected_sample_rate_hz"],
            model_sha256=value["model_sha256"],
            worker_build=value["worker_build"],
        )

    def durable_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "provider": self.provider,
            "device": self.device,
            "require_accelerator": self.require_accelerator,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "model_path": self.model_path,
            "worker_path": self.worker_path,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_sha256": self.profile_sha256,
            "expected_sample_rate_hz": self.expected_sample_rate_hz,
            "max_windows": self.max_windows,
            "max_raw_bytes": self.max_raw_bytes,
            "max_candidate_windows": self.max_candidate_windows,
            "timeout_seconds": self.timeout_seconds,
            "model_sha256": self.model_sha256,
            "worker_build": self.worker_build,
        }


def valid_durable_config(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value == {"enabled": False}:
        return True
    expected = {
        "enabled", "provider", "device", "require_accelerator",
        "allow_cpu_fallback", "model_path", "worker_path", "window_seconds",
        "hop_seconds", "profile_id", "profile_version", "profile_sha256",
        "expected_sample_rate_hz", "max_windows", "max_raw_bytes",
        "max_candidate_windows", "timeout_seconds", "model_sha256", "worker_build",
    }
    if set(value) != expected or value.get("enabled") is not True:
        return False
    if value.get("provider") != "ced" or value.get("device") not in {"metal", "cpu"}:
        return False
    if not isinstance(value.get("require_accelerator"), bool) or not isinstance(value.get("allow_cpu_fallback"), bool):
        return False
    if value["require_accelerator"] == value["allow_cpu_fallback"]:
        return False
    if value["device"] == "metal" and not value["require_accelerator"]:
        return False
    if value["device"] == "cpu" and value["require_accelerator"]:
        return False
    if (
        value.get("profile_id") != _PROFILE["id"]
        or type(value.get("profile_version")) is not int
        or value["profile_version"] != _PROFILE["version"]
    ):
        return False
    if (
        value.get("profile_sha256") != _profile_digest()
        or type(value.get("expected_sample_rate_hz")) is not int
        or value["expected_sample_rate_hz"] != _PROFILE["sample_rate_hz"]
    ):
        return False
    if not all(_bounded_string(value.get(name), limit=4096) for name in ("model_path", "worker_path", "worker_build")):
        return False
    if not _SHA256_RE.fullmatch(str(value.get("model_sha256") or "")):
        return False
    window = _strict_json_number(value.get("window_seconds"))
    hop = _strict_json_number(value.get("hop_seconds"))
    if window is None or hop is None or window <= 0 or hop <= 0 or hop > window:
        return False
    try:
        _sample_count("window_seconds", window, _PROFILE["sample_rate_hz"])
        _sample_count("hop_seconds", hop, _PROFILE["sample_rate_hz"])
    except AudioEventError:
        return False
    if not all(
        type(value.get(name)) is int and value[name] > 0
        for name in ("max_windows", "max_raw_bytes", "max_candidate_windows", "timeout_seconds")
    ):
        return False
    return (
        value["max_windows"] <= _AUDIO_EVENT_MAX_WINDOWS
        and value["max_candidate_windows"] <= _AUDIO_EVENT_MAX_EVENTS
    )


def runtime_enabled(environ: Mapping[str, Any] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return _as_bool("AUDIO_EVENTS_RUNTIME_ENABLED", env.get("AUDIO_EVENTS_RUNTIME_ENABLED", "true"))


def effective_route(settings: AudioEventSettings) -> str:
    """Return the sealed provider/device identity used by checkpoints."""
    if not settings.enabled:
        raise AudioEventError("Disabled audio-event settings have no effective route")
    return f"{settings.provider}-{settings.device}"


def source_audio_duration(
    duration_seconds: float,
    stream_start_seconds: float = 0.0,
) -> float:
    """Return the selected stream's usable source-timeline duration."""
    duration = _positive_float("source duration", duration_seconds)
    stream_start = _finite_number(stream_start_seconds)
    if stream_start is None or duration <= stream_start:
        raise AudioEventError("Selected audio stream is outside the source timeline")
    return duration - stream_start


def window_count(
    duration_seconds: float,
    settings: AudioEventSettings,
    *,
    stream_start_seconds: float = 0.0,
) -> int:
    audio_duration = source_audio_duration(duration_seconds, stream_start_seconds)
    if not settings.enabled:
        return 0
    # The worker starts one CED window at every non-empty hop position and
    # zero-pads only its final partial window. Keep preflight exactly aligned
    # with that loop so the parent never admits a job the worker must reject.
    count = max(1, int(math.ceil(audio_duration / settings.hop_seconds)))
    if count > settings.max_windows:
        raise AudioEventError(
            f"Audio-event analysis needs {count} windows, exceeding "
            f"AUDIO_EVENTS_MAX_WINDOWS={settings.max_windows}"
        )
    return count


def probe_audio_stream(ffprobe: str, source_file: str, timeout_seconds: int = 30) -> dict[str, Any] | None:
    """Select the default source audio stream or the lowest absolute index."""
    command = [
        ffprobe, "-v", "error", "-show_streams", "-of", "json", source_file,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        payload = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise AudioEventError(f"Unable to inspect source audio stream: {error}") from error
    if result.returncode != 0:
        raise AudioEventError(result.stderr.strip() or "Unable to inspect source audio stream")
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise AudioEventError("Source audio inspection returned an invalid stream list")
    candidates = []
    for stream in payload["streams"]:
        if not isinstance(stream, dict) or stream.get("codec_type") != "audio":
            continue
        index = stream.get("index")
        if not isinstance(index, int) or index < 0:
            continue
        start = stream.get("start_time")
        start_seconds = 0.0 if start in (None, "") else _finite_number(start)
        if start_seconds is None:
            raise AudioEventError("Source audio stream has an invalid start time")
        disposition = stream.get("disposition") if isinstance(stream.get("disposition"), dict) else {}
        candidates.append({
            "absolute_stream_index": index,
            "start_seconds": start_seconds,
            "selection_rule": "default" if disposition.get("default") == 1 else "lowest-index",
            "language": (stream.get("tags") or {}).get("language") if isinstance(stream.get("tags"), dict) else None,
            "title": (stream.get("tags") or {}).get("title") if isinstance(stream.get("tags"), dict) else None,
        })
    if not candidates:
        return None
    defaults = [item for item in candidates if item["selection_rule"] == "default"]
    return min(defaults or candidates, key=lambda item: item["absolute_stream_index"])


def extract_pcm_wav(
    ffmpeg: str,
    source_file: str,
    stream: Mapping[str, Any],
    output_file: str,
    sample_rate_hz: int,
    timeout_seconds: int,
) -> float:
    stream_index = stream.get("absolute_stream_index")
    if not isinstance(stream_index, int) or stream_index < 0:
        raise AudioEventError("Audio stream selection is invalid")
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", source_file, "-map", f"0:{stream_index}",
        "-vn", "-ac", "1", "-ar", str(sample_rate_hz), "-c:a", "pcm_s16le", "-n", output_file,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AudioEventError(f"Unable to extract audio for event analysis: {error}") from error
    if result.returncode != 0:
        raise AudioEventError(result.stderr.strip() or "Unable to extract audio for event analysis")
    if not os.path.isfile(output_file) or os.path.islink(output_file) or os.path.getsize(output_file) <= 44:
        raise AudioEventError("Audio-event PCM extraction did not produce a regular WAV file")
    try:
        with wave.open(output_file, "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != sample_rate_hz
                or wav.getcomptype() != "NONE"
                or wav.getnframes() <= 0
            ):
                raise AudioEventError("Audio-event PCM extraction has an unexpected WAV format")
            return wav.getnframes() / float(sample_rate_hz)
    except (OSError, wave.Error) as error:
        raise AudioEventError(f"Audio-event PCM extraction is unreadable: {error}") from error


def _validate_raw_output(
    raw: Any,
    settings: AudioEventSettings,
    source_duration: float,
    audio_stream: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
        or raw.get("provider") != "ced"
    ):
        raise AudioEventError("CED worker output has an unsupported schema")
    if raw.get("worker_build") != settings.worker_build or raw.get("model_sha256") != settings.model_sha256:
        raise AudioEventError("CED worker output does not match the sealed worker or model")
    runtime = raw.get("runtime")
    audio = raw.get("audio")
    labels = raw.get("labels")
    windows = raw.get("windows")
    if not isinstance(runtime, dict) or not isinstance(audio, dict) or not isinstance(labels, list) or not isinstance(windows, list):
        raise AudioEventError("CED worker output is missing required values")
    if runtime.get("requested_device") != settings.device:
        raise AudioEventError("CED worker used an unexpected requested device")
    observed = _bounded_string(runtime.get("observed_device"), limit=128)
    if observed is None:
        raise AudioEventError("CED worker did not report its execution device")
    if settings.require_accelerator and "MTL" not in observed.upper():
        raise AudioEventError("CED worker did not confirm the required Metal device")
    if settings.device == "cpu" and observed.lower() != "cpu":
        raise AudioEventError("CED worker did not honor the requested CPU device")
    if audio.get("sample_rate_hz") != settings.expected_sample_rate_hz:
        raise AudioEventError("CED worker used an unexpected sample rate")
    expected_window = settings.window_seconds
    expected_hop = settings.hop_seconds
    stream_start = _strict_json_number(audio.get("stream_start_seconds"))
    expected_stream_start = _strict_json_number(audio_stream.get("start_seconds"))
    if stream_start is None or expected_stream_start is None:
        raise AudioEventError("CED worker output has invalid audio stream timing")
    timing_tolerance = 1.0 / settings.expected_sample_rate_hz
    if abs(stream_start - expected_stream_start) > timing_tolerance:
        raise AudioEventError("CED worker output does not match the selected audio stream")
    decoded_duration = _strict_json_number(audio.get("decoded_duration_seconds"))
    maximum_audio_duration = source_duration - expected_stream_start
    if (
        decoded_duration is None
        or decoded_duration <= 0
        or decoded_duration > maximum_audio_duration + timing_tolerance
    ):
        raise AudioEventError("CED worker output has invalid decoded audio duration")
    expected_count = max(1, int(math.ceil(decoded_duration / expected_hop)))
    if (
        type(audio.get("window_count")) is not int
        or audio["window_count"] != len(windows)
        or len(windows) != expected_count
        or len(windows) > settings.max_windows
    ):
        raise AudioEventError("CED worker output has an invalid window count")
    if not labels or len(labels) > _AUDIO_EVENT_MAX_LABELS or any(_bounded_string(label, limit=160) is None for label in labels):
        raise AudioEventError("CED worker output has invalid labels")
    if len(set(labels)) != len(labels):
        raise AudioEventError("CED worker output has duplicate labels")
    missing_profile_labels = set(_PROFILE["labels"]).difference(labels)
    if missing_profile_labels:
        raise AudioEventError(
            "CED worker does not provide required gameplay profile label(s): "
            + ", ".join(sorted(missing_profile_labels))
        )
    if _strict_json_number(audio.get("window_seconds")) != expected_window or _strict_json_number(audio.get("hop_seconds")) != expected_hop:
        raise AudioEventError("CED worker output does not match the sealed window profile")
    normalized_windows = []
    last_start = -float("inf")
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise AudioEventError("CED worker returned an invalid window")
        start = _strict_json_number(window.get("start_seconds"))
        end = _strict_json_number(window.get("end_seconds"))
        padding = _strict_json_number(window.get("padded_tail_seconds"))
        scores = window.get("scores")
        if start is None or end is None or padding is None or start < last_start or end <= start or padding < 0:
            raise AudioEventError("CED worker returned invalid window timing")
        expected_start = stream_start + index * expected_hop
        if abs(start - expected_start) > timing_tolerance:
            raise AudioEventError("CED worker output has an unexpected window position")
        if padding > expected_window + timing_tolerance or abs(
            (end - start) + padding - expected_window
        ) > timing_tolerance:
            raise AudioEventError("CED worker output has an unexpected window length")
        if not isinstance(scores, list) or len(scores) != len(labels):
            raise AudioEventError("CED worker returned an invalid score vector")
        if any((score := _strict_json_number(value)) is None or score < 0 or score > 1 for value in scores):
            raise AudioEventError("CED worker returned an invalid score")
        # Clamp only positive intersections. The worker may report a selected
        # stream whose timestamp begins before the video timeline.
        clipped_start, clipped_end = max(0.0, start), min(source_duration, end)
        if clipped_end > clipped_start:
            normalized_windows.append({
                "start_seconds": clipped_start,
                "end_seconds": clipped_end,
                "padded_tail_seconds": padding,
                "scores": [float(score) for score in scores],
                "source_window_index": index,
            })
        last_start = start
    if abs(windows[-1]["end_seconds"] - (stream_start + decoded_duration)) > timing_tolerance:
        raise AudioEventError("CED worker output does not match decoded audio duration")
    return {
        "labels": labels,
        "windows": normalized_windows,
        "runtime": {"requested_device": settings.device, "observed_device": observed},
        "stream_start_seconds": stream_start,
    }


def _qualified_events(labels: list[str], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = _PROFILE["labels"]
    index_by_label = {label: index for index, label in enumerate(labels)}
    raw = []
    for window in windows:
        for label, threshold in thresholds.items():
            index = index_by_label.get(label)
            if index is None:
                continue
            score = window["scores"][index]
            if score >= threshold:
                raw.append({
                    "label": label,
                    "start_seconds": window["start_seconds"],
                    "end_seconds": window["end_seconds"],
                    "max_score": score,
                })
    raw.sort(key=lambda value: (value["label"], value["start_seconds"], value["end_seconds"]))
    merged = []
    for event in raw:
        if merged and event["label"] == merged[-1]["label"] and event["start_seconds"] <= merged[-1]["end_seconds"] + _PROFILE["merge_gap_seconds"]:
            merged[-1]["end_seconds"] = max(merged[-1]["end_seconds"], event["end_seconds"])
            merged[-1]["max_score"] = max(merged[-1]["max_score"], event["max_score"])
        else:
            merged.append(dict(event))
    if len(merged) > _AUDIO_EVENT_MAX_EVENTS:
        raise AudioEventError("CED result contains too many qualified events")
    return merged


def normalize_ced_output(
    raw: Any,
    settings: AudioEventSettings,
    *,
    source_identity: Mapping[str, Any],
    source_duration_seconds: float,
    audio_stream: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate native output and make the durable provider-neutral artifact."""
    duration = _positive_float("source duration", source_duration_seconds)
    parsed = _validate_raw_output(raw, settings, duration, audio_stream)
    source_sha = source_identity.get("sha256") if isinstance(source_identity, Mapping) else None
    if not _SHA256_RE.fullmatch(str(source_sha or "")):
        raise AudioEventError("Audio-event artifact needs a source SHA-256")
    events = _qualified_events(parsed["labels"], parsed["windows"])
    return {
        "schema_version": _AUDIO_EVENT_SCHEMA_VERSION,
        "provider": "ced",
        "profile": {
            "id": settings.profile_id,
            "version": settings.profile_version,
            "sha256": settings.profile_sha256,
        },
        "identity": {
            "source_sha256": source_sha,
            "model_sha256": settings.model_sha256,
            "worker_build": settings.worker_build,
            "sample_rate_hz": settings.expected_sample_rate_hz,
        },
        "runtime": parsed["runtime"],
        "audio_stream": {
            "absolute_stream_index": audio_stream.get("absolute_stream_index"),
            "start_seconds": parsed["stream_start_seconds"],
            "selection_rule": audio_stream.get("selection_rule"),
        },
        "events": events,
        "summary": {
            "qualified_event_count": len(events),
            "window_count": len(parsed["windows"]),
            "gpu_residency_note": (
                "CED device name confirms native backend selection, not "
                "per-operation residency."
            ),
        },
    }


def validate_event_artifact(value: Any, *, settings: AudioEventSettings | None = None, source_identity: Mapping[str, Any] | None = None) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {"schema_version", "provider", "profile", "identity", "runtime", "audio_stream", "events", "summary"}
    if (
        set(value) != expected
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != _AUDIO_EVENT_SCHEMA_VERSION
        or value.get("provider") != "ced"
    ):
        return False
    profile, identity, runtime, stream, events, summary = (value.get(name) for name in ("profile", "identity", "runtime", "audio_stream", "events", "summary"))
    if not all(isinstance(item, dict) for item in (profile, identity, runtime, stream, summary)) or not isinstance(events, list):
        return False
    if (
        profile.get("id") != _PROFILE["id"]
        or type(profile.get("version")) is not int
        or profile["version"] != _PROFILE["version"]
        or profile.get("sha256") != _profile_digest()
    ):
        return False
    if not _SHA256_RE.fullmatch(str(identity.get("source_sha256") or "")) or not _SHA256_RE.fullmatch(str(identity.get("model_sha256") or "")):
        return False
    if not _bounded_string(identity.get("worker_build"), limit=4096) or identity.get("sample_rate_hz") != _PROFILE["sample_rate_hz"]:
        return False
    source_duration = None
    if source_identity is not None:
        if not isinstance(source_identity, Mapping):
            return False
        if identity["source_sha256"] != source_identity.get("sha256"):
            return False
        duration_ms = _strict_json_number(source_identity.get("duration_ms"))
        if duration_ms is None or duration_ms <= 0:
            return False
        source_duration = duration_ms / 1000.0
    if settings is not None and (
        identity["model_sha256"] != settings.model_sha256
        or identity["worker_build"] != settings.worker_build
        or runtime.get("requested_device") != settings.device
        or (settings.require_accelerator and "MTL" not in str(runtime.get("observed_device")).upper())
        or (settings.device == "cpu" and str(runtime.get("observed_device")).lower() != "cpu")
    ):
        return False
    if runtime.get("requested_device") not in {"metal", "cpu"} or _bounded_string(runtime.get("observed_device"), limit=128) is None:
        return False
    if type(stream.get("absolute_stream_index")) is not int or stream["absolute_stream_index"] < 0 or _strict_json_number(stream.get("start_seconds")) is None:
        return False
    if stream.get("selection_rule") not in {"default", "lowest-index"}:
        return False
    if type(summary.get("qualified_event_count")) is not int or summary["qualified_event_count"] != len(events):
        return False
    if type(summary.get("window_count")) is not int or summary["window_count"] < 0 or summary["window_count"] > _AUDIO_EVENT_MAX_WINDOWS:
        return False
    if not _bounded_string(summary.get("gpu_residency_note"), limit=256):
        return False
    for event in events:
        if not isinstance(event, dict) or set(event) != {"label", "start_seconds", "end_seconds", "max_score"}:
            return False
        start, end, score = (_strict_json_number(event.get(name)) for name in ("start_seconds", "end_seconds", "max_score"))
        label = _bounded_string(event.get("label"), limit=160)
        threshold = _PROFILE["labels"].get(label)
        if (
            label is None
            or threshold is None
            or start is None
            or end is None
            or score is None
            or start < 0
            or end <= start
            or (source_duration is not None and end > source_duration + (1 / _PROFILE["sample_rate_hz"]))
            or score < threshold
            or score > 1
        ):
            return False
    return len(events) <= _AUDIO_EVENT_MAX_EVENTS


def build_selection_windows(
    transcript_windows: list[dict[str, Any]],
    artifact: Mapping[str, Any],
    *,
    source_duration_seconds: float,
    maximum_windows: int,
    minimum_window_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Add bounded event-only opportunities without inventing transcript text."""
    if maximum_windows <= 0:
        raise AudioEventError("Audio-event candidate window cap must be positive")
    source_duration = _positive_float("source duration", source_duration_seconds)
    minimum_window = (
        0.0
        if minimum_window_seconds is None
        else _positive_float(
            "Audio-event candidate window duration",
            minimum_window_seconds,
        )
    )
    if not validate_event_artifact(artifact):
        raise AudioEventError("Audio-event artifact is invalid")
    result = []
    for window in transcript_windows:
        entry = dict(window)
        entry["word_snapping_allowed"] = True
        evidence = event_evidence_for_range(
            artifact, entry.get("start"), entry.get("end"),
        )
        if evidence:
            entry["audio_events"] = evidence
        result.append(entry)
    padding = _PROFILE["candidate_padding_seconds"]
    for event_index, event in enumerate(sorted(artifact["events"], key=lambda item: item["max_score"], reverse=True)[:maximum_windows]):
        event_duration = event["end_seconds"] - event["start_seconds"]
        window_duration = max(
            minimum_window,
            event_duration + 2.0 * padding,
        )
        center = (event["start_seconds"] + event["end_seconds"]) / 2.0
        start = max(0.0, center - window_duration / 2.0)
        end = min(source_duration, start + window_duration)
        start = max(0.0, end - window_duration)
        if end <= start:
            continue
        result.append({
            "id": f"audio_event_{event_index + 1:03d}",
            "start": start,
            "end": end,
            "text": "[No speech transcript; probabilistic gameplay-audio evidence below.]",
            "word_snapping_allowed": False,
            "audio_events": [dict(event)],
        })
    return result


def selection_prompt_evidence(window: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = window.get("audio_events") if isinstance(window, Mapping) else None
    if not isinstance(values, list):
        return []
    evidence = []
    for item in values[:_PROFILE["max_evidence_per_candidate"]]:
        if not isinstance(item, dict):
            continue
        label = _bounded_string(item.get("label"), limit=160)
        score = _finite_number(item.get("max_score"))
        start = _finite_number(item.get("start_seconds"))
        end = _finite_number(item.get("end_seconds"))
        if label and score is not None and start is not None and end is not None:
            evidence.append({"label": label, "score": round(score, 4), "start": round(start, 3), "end": round(end, 3)})
    return evidence


def event_evidence_for_range(artifact: Mapping[str, Any], start_seconds: float, end_seconds: float) -> list[dict[str, Any]]:
    start, end = _finite_number(start_seconds), _finite_number(end_seconds)
    if start is None or end is None or end <= start or not validate_event_artifact(artifact):
        return []
    overlaps = [
        event for event in artifact["events"]
        if event["end_seconds"] > start and event["start_seconds"] < end
    ]
    return selection_prompt_evidence({"audio_events": sorted(overlaps, key=lambda item: item["max_score"], reverse=True)})
