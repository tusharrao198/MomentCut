"""Versioned, integrity-checked compute checkpoint storage for OpenShorts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import job_state


class CheckpointError(RuntimeError):
    pass


SCHEMA_VERSION = 1
MAX_CHECKPOINT_FILE_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_PROVENANCE_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIXED_STAGES = {
    "source": "source.json",
    "asr": "asr.json",
    "source-layout": "source-layout.json",
    "source-subtitles": "source-subtitles.json",
    "source-scenes": "source-scenes.json",
    "text-rank": "text-rank.json",
    "cloud-visual": "cloud-visual.json",
    "clip-plan": "clip-plan.json",
}
_KEYED_STAGES = {
    "text-score": "text-score",
    "text-detail": "text-detail",
    "visual-detail": "visual-detail",
    "render": "render",
}
_IMPLEMENTATION_VERSIONS = {
    "source": "source-v1",
    "asr": "asr-v1",
    "source-layout": "source-layout-v1",
    "source-subtitles": "source-subtitles-v1",
    "source-scenes": "source-scenes-v1",
    "text-score": "text-score-v1",
    "text-detail": "text-detail-v1",
    "text-rank": "text-rank-v1",
    "visual-detail": "visual-detail-v1",
    "cloud-visual": "cloud-visual-v1",
    "clip-plan": "clip-plan-v1",
    "render": "render-v1",
}
ASR_ROUTES = frozenset({
    "parakeet", "parakeet-mlx", "parakeet-to-whisper-gpu",
    "parakeet-to-whisper-cpu", "parakeet-to-whisper-mlx", "whisper-gpu",
    "whisper-cpu", "whisper-mlx", "no-audio", "precomputed-transcript",
})
ASR_OUTCOMES = frozenset({"speech", "no_audio", "sparse_speech"})
SELECTION_ROUTES = frozenset({
    "speech-local", "speech-gemini", "silent-local", "silent-gemini",
})

# Non-secret environment values that can alter source processing, model
# decisions, or rendered output. app.py persists these at acceptance and
# reconstructs its child environment from this list; standalone main.py records
# the same surface so changed settings are detected before recovery.
PIPELINE_ENV_NAMES = (
    "AUTO_LAYOUT", "SPLIT_LAYOUT", "SCREENCAST_LAYOUT", "CAMERA_INSET",
    "PANEL_LAYOUT", "SPEAKER_SIGNAL", "SPEAKER_CUT", "PUNCH_IN",
    "LAYOUT_SAMPLE_FRAMES", "LAYOUT_SAMPLE_WIDTH", "SUBTITLE_BAND_CHECK",
    "SPLIT_TIGHTNESS", "PANEL_TIGHTNESS", "SPLIT_MIN_SHARE",
    "INSET_PADDING", "SCREENCAST_MAX_SHOTS", "PUNCH_IN_ZOOM",
    "HOOK_GROUNDING", "HOOK_GROUNDING_FRAMES", "HOOK_GROUNDING_WIDTH",
    "HOOK_GROUNDING_MIN_SHARE",
    "AUTO_HOOK", "AUTO_HOOK_STYLE", "AUTO_HOOK_SECONDS", "AUTO_CAPTIONS",
    "WATERMARK", "CLIP_TARGET_MIN", "CLIP_TARGET_MAX", "CLIP_MIN_SECONDS",
    "CLIP_MAX_SECONDS", "GEMINI_MODEL", "GEMINI_THINKING_SCORE",
    "GEMINI_CLIP_ANALYSIS_ENABLED",
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT",
    "LLM_SCORE_BATCH", "LLM_DETAIL_FALLBACK_WINDOWS", "VISION_PROVIDER",
    "VISION_BASE_URL", "VISION_MODEL", "VISION_TIMEOUT_SECONDS",
    "VISION_RETRY_ATTEMPTS", "VISION_WINDOW_SECONDS",
    "VISION_WINDOW_OVERLAP_SECONDS", "VISION_FRAMES_PER_WINDOW",
    "VISION_MAX_FRAME_WIDTH", "VISION_NUM_CTX",
    "VISION_CANDIDATES_PER_WINDOW", "VISION_SCENE_SNAP_SECONDS",
    "TRANSCRIBE_BACKEND", "WHISPER_MODEL", "WHISPER_MODEL_REVISION",
    "WHISPER_DEVICE", "WHISPER_COMPUTE", "WHISPER_ENGINE",
    "WHISPER_MLX_MODEL", "WHISPER_MLX_MODEL_REVISION", "PARAKEET_MODEL_ID",
    "PARAKEET_MODEL_REVISION", "PARAKEET_VAD_BATCH", "PARAKEET_ENGINE",
    "PARAKEET_MLX_MODEL", "PARAKEET_MLX_MODEL_REVISION", "MIN_SPEECH_WORDS",
    "MIN_SPEECH_WORDS_PER_MIN", "APPLE_SILICON_ACCELERATION", "SCENE_ENGINE",
    "TRANSNETV2_DEVICE", "TRANSNETV2_MODEL_REVISION", "TRANSNETV2_THRESHOLD",
    "GPU_DECODE_MIN_FREE_MB",
    "SCENE_MIN_SEC", "VIDEO_DECODE_BACKEND", "GPU_DECODE", "GPU_DECODE_RENDER",
    "LLM_THINKING",
    "FFMPEG_ENCODER", "FFMPEG_GPU_CUT", "CUT_NVENC_PRESET",
    "CUT_NVENC_CQ", "AUDIO_NORMALIZE", "REFRAME_ENGINE",
    "GENERAL_CONTENT_HEIGHT_RATIO", "SPARSE_TRAJECTORY", "YOLO_MODEL_PATH",
    "DETECT_STRIDE", "JUMP_CONFIRM_FRAMES", "SCENE_CUT_RESET",
    "CLIP_WORKERS", "CLIP_RETRY_PAUSE_SECONDS", "EARLY_AUDIO",
    "MAX_SOURCE_MINUTES", "SOURCE_CAP_MINUTES", "DOWNLOAD_SKIP_STATICS",
)


def pipeline_checkpoints_enabled(env: dict | None = None) -> bool:
    source = os.environ if env is None else env
    raw = source.get("PIPELINE_CHECKPOINTS_ENABLED")
    if raw is None or str(raw).strip() == "":
        return False
    value = str(raw).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise CheckpointError(
        "PIPELINE_CHECKPOINTS_ENABLED must be exactly true or false "
        "(case-insensitive)"
    )


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError("Checkpoint JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError("Checkpoint object keys must be strings")
            key = unicodedata.normalize("NFC", key)
            if key in result:
                raise CheckpointError(
                    "Checkpoint object has duplicate keys after NFC normalization"
                )
            result[key] = _normalise(item)
        return result
    raise CheckpointError(f"Unsupported checkpoint value: {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        _normalise(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_key(stage: str, inputs: dict) -> str:
    if stage not in _IMPLEMENTATION_VERSIONS:
        raise CheckpointError(f"Unsupported checkpoint stage: {stage}")
    return sha256_bytes(canonical_json_bytes({
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        "implementation_version": _IMPLEMENTATION_VERSIONS[stage],
        **inputs,
    }))


def _content_identity(source_identity: dict) -> dict:
    if not isinstance(source_identity, dict):
        raise CheckpointError("Source identity is required")
    sha = source_identity.get("sha256")
    size = source_identity.get("size")
    if not _SHA256_RE.fullmatch(str(sha or "")) or not isinstance(size, int) or size < 0:
        raise CheckpointError("Source identity lacks a valid SHA-256 and size")
    return {"sha256": sha, "size": size}


def _stage_config(job_config: dict, *names: str) -> dict:
    result = {}
    for name in names:
        if name in job_config:
            result[name] = job_config[name]
    result["configuration_revision"] = job_config.get("configuration_revision")
    return result


def expected_source_request_key(job_config: dict) -> str:
    source = dict(job_config.get("source") or {})
    source.pop("original_name", None)
    return compatibility_key("source", {
        "source_content_identity": None,
        "route": source.get("kind"),
        "resolved_stage_config": source,
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": [],
    })


def expected_asr_key(
    job_config: dict,
    source_identity: dict,
    effective_route: str,
) -> str:
    return compatibility_key("asr", {
        "source_content_identity": _content_identity(source_identity),
        "route": effective_route,
        "resolved_stage_config": _stage_config(job_config, "asr"),
        "expected_upstream_compatibility_keys": [
            expected_source_request_key(job_config),
        ],
        "semantic_input_hashes": [],
    })


def expected_clip_plan_key(
    job_config: dict,
    source_identity: dict,
    route_context: dict,
) -> str:
    if not isinstance(route_context, dict):
        raise CheckpointError("Clip-plan route context must be an object")
    upstream = [expected_source_request_key(job_config)]
    asr_route = route_context.get("asr_effective_route")
    if asr_route:
        upstream.append(expected_asr_key(job_config, source_identity, asr_route))
    return compatibility_key("clip-plan", {
        "source_content_identity": _content_identity(source_identity),
        "route": route_context,
        "resolved_stage_config": _stage_config(
            job_config, "pipeline", "selection", "asr", "scene_detection",
            "text_selection", "vision_selection",
        ),
        "expected_upstream_compatibility_keys": upstream,
        "semantic_input_hashes": [],
    })


def expected_render_key(
    job_config: dict,
    source_identity: dict,
    clip_plan_envelope: dict,
    clip: dict,
    source_analysis: dict,
) -> str:
    plan_key = clip_plan_envelope.get("compatibility_key")
    plan_hash = clip_plan_envelope.get("payload_hash")
    if not _SHA256_RE.fullmatch(str(plan_key or "")):
        raise CheckpointError("Clip-plan envelope lacks a valid compatibility key")
    return compatibility_key("render", {
        "source_content_identity": _content_identity(source_identity),
        "route": "render",
        "resolved_stage_config": _stage_config(job_config, "pipeline", "render"),
        "expected_upstream_compatibility_keys": [plan_key],
        "semantic_input_hashes": [
            value for value in [
                plan_hash,
                sha256_bytes(canonical_json_bytes(clip)),
                sha256_bytes(canonical_json_bytes(source_analysis or {})),
            ] if value
        ],
    })


def build_source_identity(path: str, probe: dict) -> dict:
    path = os.path.abspath(os.fspath(path))
    result = {
        "sha256": sha256_file(path),
        "size": os.path.getsize(path),
        "duration_ms": int(round(float((probe or {}).get("duration", 0)) * 1000)),
    }
    fps = (probe or {}).get("fps")
    if fps is not None:
        result["fps"] = float(fps)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _same_source(expected: dict, actual: dict) -> bool:
    try:
        if _content_identity(expected) != _content_identity(actual):
            return False
        expected_ms = int(expected.get("duration_ms", 0))
        actual_ms = int(actual.get("duration_ms", 0))
        fps_values = [
            float(value) for value in (expected.get("fps"), actual.get("fps"))
            if value and float(value) > 0
        ]
        frame_tolerance = max((2000.0 / fps for fps in fps_values), default=0)
        return abs(expected_ms - actual_ms) <= max(500, frame_tolerance)
    except (TypeError, ValueError, CheckpointError):
        return False


def _valid_transcript(
    transcript: Any,
    *,
    source_duration: float | None,
    require_segments: bool,
) -> bool:
    if not isinstance(transcript, dict):
        return False
    if not isinstance(transcript.get("language"), str):
        return False
    segments = transcript.get("segments")
    if not isinstance(segments, list) or (require_segments and not segments):
        return False
    last_end = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
            # ASR backends can produce tiny boundary overlaps between adjacent
            # segments. They are still chronological and safe to reuse; reject
            # only material backwards movement.
            or start < last_end - 0.5
            or (
                source_duration is not None
                and end > source_duration + 0.5
            )
            or not isinstance(segment.get("text", ""), str)
        ):
            return False
        words = segment.get("words", [])
        if not isinstance(words, list):
            return False
        last_word_end = start
        for word in words:
            if not isinstance(word, dict) or not isinstance(
                word.get("word", ""), str
            ):
                return False
            try:
                word_start = float(word["start"])
                word_end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                return False
            if (
                not math.isfinite(word_start)
                or not math.isfinite(word_end)
                or word_start < start - 0.5
                or word_end < word_start
                or word_start < last_word_end - 0.5
                or word_end > end + 0.5
            ):
                return False
            last_word_end = word_end
        last_end = end
    return True


def validate_asr_route(value: Any) -> str:
    if not isinstance(value, str) or value not in ASR_ROUTES:
        raise CheckpointError("Unknown ASR effective route")
    return value


def valid_asr_payload(
    payload: Any,
    source_duration: float | None = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    outcome = payload.get("outcome")
    transcript = payload.get("transcript")
    if outcome == "no_audio":
        return "transcript" in payload and transcript is None and set(payload).issubset(
            {"outcome", "transcript"}
        )
    if outcome not in {"speech", "sparse_speech"}:
        return False
    if not _valid_transcript(
        transcript,
        source_duration=source_duration,
        require_segments=outcome == "speech",
    ):
        return False
    if outcome == "sparse_speech":
        count = sum(
            len((segment.get("text") or "").split())
            for segment in transcript["segments"]
        )
        if payload.get("word_count", count) != count:
            return False
    return True


def _valid_artifact_descriptor(descriptor: Any) -> bool:
    if not isinstance(descriptor, dict):
        return False
    relative = descriptor.get("relative_path") or descriptor.get("filename")
    probe = descriptor.get("probe")
    if (
        not isinstance(relative, str)
        or not relative
        or os.path.isabs(relative)
        or os.path.basename(relative) != relative
        or relative.startswith(".")
        or not isinstance(descriptor.get("size"), int)
        or descriptor["size"] <= 0
        or not isinstance(descriptor.get("sha256"), str)
        or not _SHA256_RE.fullmatch(descriptor["sha256"])
        or not isinstance(descriptor.get("duration_ms"), int)
        or descriptor["duration_ms"] <= 0
        or not isinstance(probe, dict)
    ):
        return False
    playable = probe.get("playable")
    duration = probe.get("duration")
    if playable is not True:
        try:
            if not math.isfinite(float(duration)) or float(duration) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _valid_render_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if (
        not isinstance(payload.get("clip_identity"), str)
        or not _SHA256_RE.fullmatch(payload["clip_identity"])
        or not isinstance(payload.get("display_index"), int)
        or payload["display_index"] < 0
        or not _valid_artifact_descriptor(payload.get("artifact"))
    ):
        return False
    metadata = payload.get("clip_metadata", {})
    return isinstance(metadata, dict)


def artifact_matches_clip_duration(
    descriptor: dict,
    clip: dict,
    *,
    tolerance_ms: int = 1500,
) -> bool:
    """Whether an artifact duration matches the immutable planned cut."""
    try:
        start = float(clip["start"])
        end = float(clip["end"])
        actual_ms = int(descriptor["duration_ms"])
        tolerance_ms = max(0, int(tolerance_ms))
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
        or actual_ms <= 0
    ):
        return False
    expected_ms = int(round((end - start) * 1000))
    return abs(actual_ms - expected_ms) <= tolerance_ms


def _valid_source_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    fingerprint = payload.get("source_fingerprint")
    artifact = payload.get("artifact")
    try:
        identity = _content_identity(fingerprint)
        duration_ms = int(fingerprint.get("duration_ms"))
    except (AttributeError, TypeError, ValueError, CheckpointError):
        return False
    if identity["size"] <= 0 or duration_ms <= 0:
        return False
    if not _valid_artifact_descriptor(artifact):
        return False
    if (
        artifact.get("size") != identity["size"]
        or artifact.get("sha256") != identity["sha256"]
        or abs(int(artifact.get("duration_ms")) - duration_ms) > 500
        or artifact.get("relative_path")
        != payload.get("retained_relative_path")
    ):
        return False
    probe = payload.get("probe")
    if not isinstance(probe, dict):
        return False
    try:
        duration = float(probe.get("duration"))
        width = int(probe.get("width"))
        height = int(probe.get("height"))
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(duration)
        and duration > 0
        and width > 0
        and height > 0
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_hash_list(value: Any, *, maximum_items: int = 4096) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= maximum_items
        and all(
            isinstance(item, str) and bool(_SHA256_RE.fullmatch(item))
            for item in value
        )
    )


def _valid_clip_candidate(
    value: Any,
    *,
    source_identity: dict | None,
    require_window_id: bool = False,
) -> bool:
    if not isinstance(value, dict):
        return False
    start = _finite_number(value.get("start"))
    end = _finite_number(value.get("end"))
    if start is None or end is None or start < 0 or end <= start:
        return False
    source_duration = _finite_number(
        (source_identity or {}).get("duration_ms")
    )
    if source_duration is not None \
            and end > source_duration / 1000.0 + 0.5:
        return False
    if require_window_id and (
        not isinstance(value.get("source_window_id"), str)
        or not value["source_window_id"]
        or len(value["source_window_id"]) > 256
    ):
        return False
    score = value.get("predicted_score")
    if score is not None:
        score = _finite_number(score)
        if score is None or score < 0 or score > 100:
            return False
    for field, maximum in (
        ("video_title_for_youtube_short", 1000),
        ("video_description_for_youtube", 20_000),
        ("youtube_description_intro", 20_000),
        ("video_description_for_tiktok", 5000),
        ("video_description_for_instagram", 5000),
        ("viral_hook_text", 2000),
        ("why", 5000),
        ("global_rank_reason", 5000),
    ):
        field_value = value.get(field)
        if field_value is not None and (
            not isinstance(field_value, str) or len(field_value) > maximum
        ):
            return False
    for field in ("youtube_tags", "youtube_hashtags"):
        field_value = value.get(field)
        if field_value is not None and (
            not isinstance(field_value, list)
            or len(field_value) > 100
            or any(
                not isinstance(item, str) or len(item) > 500
                for item in field_value
            )
        ):
            return False
    thumbnail = value.get("thumbnail_brief")
    if thumbnail is not None:
        try:
            if not isinstance(thumbnail, dict) \
                    or len(canonical_json_bytes(thumbnail)) > 64 * 1024:
                return False
        except CheckpointError:
            return False
    return True


def _valid_source_layout_payload(payload: dict) -> bool:
    allowed_flags = {"split_layout", "screencast_layout", "active_speaker"}
    flags = payload.get("applied_flags")
    return (
        payload.get("outcome") == "complete"
        and payload.get("layout") in {"none", "screencast", "split"}
        and isinstance(flags, list)
        and len(flags) <= len(allowed_flags)
        and len(set(flags)) == len(flags)
        and all(flag in allowed_flags for flag in flags)
        and _valid_hash_list(payload.get("sample_hashes"), maximum_items=128)
        and isinstance(payload.get("config"), dict)
        and payload.get("implementation_version") == "source-layout-v1"
    )


def _valid_source_subtitles_payload(payload: dict) -> bool:
    outcome = payload.get("outcome")
    band = payload.get("band")
    if outcome == "none":
        if band is not None:
            return False
    elif outcome == "band":
        band = _finite_number(band)
        if band is None or not 0 < band < 1:
            return False
    else:
        return False
    return (
        _valid_hash_list(payload.get("sample_hashes"), maximum_items=128)
        and isinstance(payload.get("config"), dict)
        and payload.get("implementation_version") == "source-subtitles-v1"
    )


def _valid_source_scenes_payload(
    payload: dict,
    source_identity: dict | None,
) -> bool:
    boundaries = payload.get("boundaries_seconds")
    fps = _finite_number(payload.get("fps"))
    detector = payload.get("detector")
    if (
        not isinstance(boundaries, list)
        or not boundaries
        or len(boundaries) > 1_000_000
        or fps is None
        or fps <= 0
        or not isinstance(detector, dict)
        or detector.get("effective_engine")
        not in {"transnetv2", "pyscenedetect"}
    ):
        return False
    duration_ms = _finite_number((source_identity or {}).get("duration_ms"))
    source_duration = (
        duration_ms / 1000.0 if duration_ms is not None else None
    )
    previous = -1.0
    for raw in boundaries:
        boundary = _finite_number(raw)
        if (
            boundary is None
            or boundary < 0
            or boundary <= previous
            or (
                source_duration is not None
                and boundary > source_duration + 0.5
            )
        ):
            return False
        previous = boundary
    return True


def _valid_embedded_source_analysis(value: Any) -> bool:
    """Validate source-level render decisions embedded in a clip plan.

    These decisions travel with the final plan so a restart cannot repeat a
    best-effort provider call and render that plan with different layout or
    subtitle-crop behavior.
    """
    if not isinstance(value, dict):
        return False
    if not {
        "source_subtitle_band", "layout_config", "checkpoints",
    }.issubset(value):
        return False
    try:
        if len(canonical_json_bytes(value)) > MAX_CHECKPOINT_PROVENANCE_BYTES:
            return False
    except CheckpointError:
        return False
    layout_config = value.get("layout_config")
    if (
        not isinstance(layout_config, list)
        or len(layout_config) > 32
        or any(
            not isinstance(name, str) or not name or len(name) > 128
            for name in layout_config
        )
    ):
        return False
    source_band = value.get("source_subtitle_band")
    if source_band is not None:
        source_band = _finite_number(source_band)
        if source_band is None or not 0 < source_band < 1:
            return False
    checkpoints = value.get("checkpoints", {})
    if not isinstance(checkpoints, dict):
        return False
    subtitle_result = None
    for name, record in checkpoints.items():
        if name not in {"layout", "subtitles"} or not isinstance(record, dict):
            return False
        key = record.get("compatibility_key")
        if not isinstance(key, str) or not _SHA256_RE.fullmatch(key):
            return False
        result = record.get("result")
        if not isinstance(result, dict):
            return False
        try:
            if len(canonical_json_bytes(result)) > MAX_CHECKPOINT_PROVENANCE_BYTES:
                return False
        except CheckpointError:
            return False
        if name == "layout":
            allowed_flags = {
                "split_layout", "screencast_layout", "active_speaker",
            }
            applied_flags = result.get("applied_flags")
            if (
                result.get("outcome") not in {"complete", "failure"}
                or result.get("layout") not in {"none", "screencast", "split"}
                or not isinstance(applied_flags, list)
                or len(applied_flags) > len(allowed_flags)
                or len(set(applied_flags)) != len(applied_flags)
                or any(flag not in allowed_flags for flag in applied_flags)
            ):
                return False
        else:
            subtitle_result = result
            if (
                result.get("outcome") not in {"band", "none", "failure"}
                or (
                    result.get("band") is not None
                    and (
                        _finite_number(result.get("band")) is None
                        or not 0 < float(result["band"]) < 1
                    )
                )
            ):
                return False
    if subtitle_result is None:
        return source_band is None
    if subtitle_result.get("outcome") == "band":
        result_band = _finite_number(subtitle_result.get("band"))
        return (
            source_band is not None
            and result_band is not None
            and abs(source_band - result_band) < 1e-9
        )
    return source_band is None and subtitle_result.get("band") is None


def _valid_text_score_payload(payload: dict) -> bool:
    score = payload.get("score")
    if (
        payload.get("origin") not in {"model", "reconciled_fallback"}
        or not isinstance(score, dict)
        or not isinstance(score.get("id"), str)
        or not score["id"]
        or len(score["id"]) > 256
    ):
        return False
    start = _finite_number(score.get("start"))
    end = _finite_number(score.get("end"))
    value = _finite_number(score.get("score"))
    return (
        start is not None
        and end is not None
        and 0 <= start < end
        and value is not None
        and 0 <= value <= 100
        and isinstance(score.get("text", ""), str)
        and isinstance(score.get("reason", ""), str)
    )


def _valid_text_detail_payload(
    payload: dict,
    source_identity: dict | None,
) -> bool:
    if payload.get("outcome") == "empty":
        return set(payload) == {"outcome"}
    return (
        payload.get("outcome") == "candidate"
        and set(payload) == {"outcome", "candidate"}
        and _valid_clip_candidate(
            payload.get("candidate"),
            source_identity=source_identity,
            require_window_id=True,
        )
    )


def _valid_text_rank_payload(payload: dict) -> bool:
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or len(rankings) > 100:
        return False
    seen = set()
    for ranking in rankings:
        if not isinstance(ranking, dict):
            return False
        candidate_id = ranking.get("candidate_id")
        score = _finite_number(ranking.get("score"))
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or len(candidate_id) > 256
            or candidate_id in seen
            or score is None
            or score < 0
            or score > 100
            or not isinstance(ranking.get("reason", ""), str)
        ):
            return False
        seen.add(candidate_id)
    return True


def _valid_visual_detail_payload(
    payload: dict,
    source_identity: dict | None,
) -> bool:
    if payload.get("outcome") == "empty":
        return set(payload) == {"outcome"}
    candidates = payload.get("candidates")
    return (
        payload.get("outcome") == "candidates"
        and set(payload) == {"outcome", "candidates"}
        and isinstance(candidates, list)
        and 0 < len(candidates) <= 100
        and all(
            _valid_clip_candidate(
                candidate,
                source_identity=source_identity,
            )
            for candidate in candidates
        )
    )


def _valid_cloud_visual_payload(
    payload: dict,
    source_identity: dict | None,
) -> bool:
    shorts = payload.get("shorts")
    return (
        isinstance(shorts, list)
        and 0 < len(shorts) <= 100
        and all(
            _valid_clip_candidate(
                candidate,
                source_identity=source_identity,
            )
            for candidate in shorts
        )
    )


def _valid_stage_payload(
    stage: str,
    payload: Any,
    source_identity: dict | None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if stage == "source":
        return _valid_source_payload(payload)
    if stage == "asr":
        source_duration = None
        try:
            source_duration = float(
                (source_identity or {}).get("duration_ms")
            ) / 1000.0
        except (TypeError, ValueError):
            pass
        return valid_asr_payload(payload, source_duration)
    if stage == "source-layout":
        return _valid_source_layout_payload(payload)
    if stage == "source-subtitles":
        return _valid_source_subtitles_payload(payload)
    if stage == "source-scenes":
        return _valid_source_scenes_payload(payload, source_identity)
    if stage == "text-score":
        return _valid_text_score_payload(payload)
    if stage == "text-detail":
        return _valid_text_detail_payload(payload, source_identity)
    if stage == "text-rank":
        return _valid_text_rank_payload(payload)
    if stage == "visual-detail":
        return _valid_visual_detail_payload(payload, source_identity)
    if stage == "cloud-visual":
        return _valid_cloud_visual_payload(payload, source_identity)
    if stage == "render":
        return _valid_render_payload(payload)
    if stage == "clip-plan":
        # Clip-plan route/transcript validation is performed by the store,
        # where the routed provenance is available.
        return True
    # A registered stage must never become reusable merely because its payload
    # happens to be a JSON object. New stages need an explicit validator.
    return False


class CheckpointStore:
    def __init__(
        self,
        project_dir: str,
        job_config: dict,
        enabled: bool,
        execution_generation: int,
        source_identity: dict | None = None,
    ):
        self.project_dir = os.path.realpath(os.path.abspath(project_dir))
        if not os.path.isdir(self.project_dir):
            raise CheckpointError("Checkpoint project directory does not exist")
        self.job_config = job_config
        # Callers pass the already-resolved deployment/job enablement. The
        # immutable job setting prevents retroactive checkpointing of a job
        # originally accepted while the feature was disabled.
        self.enabled = bool(enabled)
        self.execution_generation = int(execution_generation)
        self.job_id = str(job_config.get("job_id") or "")
        self.configuration_revision = int(
            job_config.get("configuration_revision") or 0
        )
        if not self.job_id or self.configuration_revision < 1:
            raise CheckpointError("Job configuration lacks durable identity")
        unhashed_config = dict(job_config)
        unhashed_config.pop("job_config_hash", None)
        self.job_config_hash = (
            job_config.get("job_config_hash")
            or sha256_bytes(canonical_json_bytes(unhashed_config))
        )
        self.root = os.path.join(self.project_dir, ".checkpoints", "v1")
        self.source_identity = None
        if source_identity is not None:
            self.bind_source(source_identity)

    def _assert_safe_root(self) -> None:
        """Reject checkpoint directories redirected outside the project."""
        lexical_root = os.path.abspath(self.root)
        if os.path.commonpath([self.project_dir, lexical_root]) != self.project_dir:
            raise CheckpointError("Checkpoint root escaped its project")
        current = self.project_dir
        for component in (".checkpoints", "v1"):
            current = os.path.join(current, component)
            if os.path.lexists(current):
                if os.path.islink(current):
                    raise CheckpointError(
                        "Checkpoint directory must not be a symlink"
                    )
                if not os.path.isdir(current):
                    raise CheckpointError(
                        "Checkpoint path component is not a directory"
                    )
        resolved_root = os.path.realpath(lexical_root)
        if os.path.commonpath(
            [self.project_dir, resolved_root]
        ) != self.project_dir:
            raise CheckpointError("Checkpoint root resolved outside its project")

    def _path(self, stage: str, expected_key: str | None = None) -> str:
        self._assert_safe_root()
        if stage in _FIXED_STAGES:
            if expected_key is not None:
                raise CheckpointError("Fixed checkpoint does not take a filename key")
            path = os.path.join(self.root, _FIXED_STAGES[stage])
        elif stage in _KEYED_STAGES:
            if not _SHA256_RE.fullmatch(str(expected_key or "")):
                raise CheckpointError("Keyed checkpoint requires a lowercase SHA-256 key")
            path = os.path.join(self.root, _KEYED_STAGES[stage], f"{expected_key}.json")
        else:
            raise CheckpointError(f"Unsupported checkpoint stage: {stage}")
        root = os.path.realpath(self.root)
        parent = os.path.realpath(os.path.dirname(path))
        if os.path.commonpath([root, parent]) != root:
            raise CheckpointError("Checkpoint path escaped its project")
        return path

    def _read(self, path: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            self._assert_safe_root()
            if os.path.islink(path):
                return None
        except CheckpointError:
            return None
        return job_state.read_json(
            path, max_bytes=MAX_CHECKPOINT_FILE_BYTES,
        )

    def _validate(
        self,
        envelope: dict,
        stage: str,
        expected_key: str,
        *,
        source_optional: bool = False,
    ) -> bool:
        try:
            if envelope.get("schema_version") != SCHEMA_VERSION:
                return False
            if envelope.get("implementation_version") != _IMPLEMENTATION_VERSIONS[stage]:
                return False
            if envelope.get("stage") != stage or envelope.get("status") != "complete":
                return False
            if envelope.get("job_id") != self.job_id:
                return False
            if envelope.get("configuration_revision") != self.configuration_revision:
                return False
            if envelope.get("compatibility_key") != expected_key:
                return False
            if envelope.get("config_hash") != self.job_config_hash:
                return False
            writer_generation = envelope.get("writer_generation")
            if (
                not isinstance(writer_generation, int)
                or isinstance(writer_generation, bool)
                or writer_generation < 0
            ):
                return False
            created_at = envelope.get("created_at")
            if not isinstance(created_at, str) or not created_at \
                    or len(created_at) > 64:
                return False
            for field in (
                "upstream_compatibility_keys",
                "semantic_input_hashes",
                "provenance_hashes",
            ):
                values = envelope.get(field)
                if (
                    not isinstance(values, list)
                    or len(values) > 4096
                    or not all(
                    isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))
                    for value in values
                    )
                ):
                    return False
            provenance = envelope.get("provenance")
            if not isinstance(provenance, dict) or len(
                canonical_json_bytes(provenance)
            ) > MAX_CHECKPOINT_PROVENANCE_BYTES:
                return False
            if envelope.get("payload_hash") != sha256_bytes(
                canonical_json_bytes(envelope.get("payload"))
            ):
                return False
            if not _valid_stage_payload(
                stage, envelope.get("payload"), self.source_identity
            ):
                return False
            if not source_optional:
                if self.source_identity is None:
                    return False
                if not _same_source(
                    self.source_identity, envelope.get("source_fingerprint") or {}
                ):
                    return False
            return True
        except Exception:
            return False

    def _envelope(
        self,
        stage: str,
        expected_key: str,
        payload: dict,
        provenance: dict,
        *,
        source_identity: dict | None = None,
    ) -> dict:
        if not isinstance(payload, dict):
            raise CheckpointError("Checkpoint payload must be an object")
        provenance = provenance or {}
        return {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": _IMPLEMENTATION_VERSIONS[stage],
            "stage": stage,
            "job_id": self.job_id,
            "configuration_revision": self.configuration_revision,
            "compatibility_key": expected_key,
            "status": "complete",
            "created_at": _utc_now(),
            "writer_generation": self.execution_generation,
            "source_fingerprint": source_identity or self.source_identity,
            "upstream_compatibility_keys": list(
                provenance.get("upstream_compatibility_keys") or []
            ),
            "semantic_input_hashes": list(
                provenance.get("semantic_input_hashes") or []
            ),
            "config_hash": self.job_config_hash,
            "payload_hash": sha256_bytes(canonical_json_bytes(payload)),
            "provenance_hashes": list(provenance.get("provenance_hashes") or []),
            "route_context": provenance.get("route_context"),
            "provenance": {
                key: value for key, value in provenance.items()
                if key not in {
                    "upstream_compatibility_keys", "semantic_input_hashes",
                    "config_hash", "provenance_hashes", "route_context",
                }
            },
            "payload": payload,
        }

    def _save(self, path: str, envelope: dict) -> bool:
        if not self.enabled:
            return False
        try:
            stage = envelope.get("stage")
            if stage not in _IMPLEMENTATION_VERSIONS or not self._validate(
                envelope,
                stage,
                envelope.get("compatibility_key"),
                source_optional=stage == "source",
            ):
                raise CheckpointError(
                    f"Refusing to persist invalid {stage or 'unknown'} checkpoint"
                )
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            job_state.atomic_write_json(path, envelope)
            return True
        except Exception as error:
            print(
                f"⚠️ Could not save {envelope.get('stage')} checkpoint: "
                f"{type(error).__name__}: {error}"
            )
            return False

    def load_source(self, expected_request_key: str) -> dict | None:
        envelope = self._read(self._path("source"))
        if envelope is None:
            return None
        fingerprint = envelope.get("source_fingerprint") or {}
        if not self._validate(
            envelope, "source", expected_request_key, source_optional=True,
        ):
            return None
        if not _same_source(fingerprint, (envelope.get("payload") or {}).get(
            "source_fingerprint", fingerprint
        )):
            return None
        descriptor = dict((envelope.get("payload") or {}).get("artifact") or {})
        if not descriptor or not self.validate_artifact(descriptor):
            return None
        return envelope

    def save_source(
        self,
        expected_request_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        fingerprint = payload.get("source_fingerprint")
        _content_identity(fingerprint)
        envelope = self._envelope(
            "source", expected_request_key, payload, provenance,
            source_identity=fingerprint,
        )
        return self._save(self._path("source"), envelope)

    def bind_source(self, source_identity: dict) -> None:
        _content_identity(source_identity)
        if self.source_identity is not None and not _same_source(
            self.source_identity, source_identity
        ):
            raise CheckpointError("Checkpoint store is already bound to another source")
        self.source_identity = dict(source_identity)

    def load_fixed(self, stage: str, expected_key: str) -> dict | None:
        if self.source_identity is None:
            raise CheckpointError("Bind the final source before loading checkpoints")
        envelope = self._read(self._path(stage))
        if envelope is None or not self._validate(envelope, stage, expected_key):
            return None
        return envelope

    def load_routed_fixed(
        self,
        stage: str,
        validate_route_context,
        expected_key_for_route,
    ) -> dict | None:
        if self.source_identity is None:
            raise CheckpointError("Bind the final source before loading checkpoints")
        envelope = self._read(self._path(stage))
        if envelope is None:
            return None
        try:
            route = validate_route_context(envelope.get("route_context"))
            expected = expected_key_for_route(route)
        except Exception:
            return None
        if not self._validate(envelope, stage, expected):
            return None
        if stage == "asr" and not self._valid_asr_provenance(envelope, route):
            return None
        return envelope

    def _valid_asr_provenance(self, envelope: dict, route: str) -> bool:
        payload = envelope.get("payload") or {}
        outcome = payload.get("outcome")
        backend = (envelope.get("provenance") or {}).get("backend")
        if route == "no-audio":
            return (
                outcome == "no_audio"
                and isinstance(backend, dict)
                and backend.get("effective_route") == route
                and backend.get("effective_backend") == "none"
            )
        if outcome == "no_audio" or not isinstance(backend, dict):
            return False
        parakeet_routes = {"parakeet", "parakeet-mlx"}
        expected_backend = (
            "external"
            if route == "precomputed-transcript"
            else "parakeet" if route in parakeet_routes else "whisper"
        )
        if (
            backend.get("effective_route") != route
            or backend.get("effective_backend") != expected_backend
        ):
            return False
        immutable = backend.get("immutable_revision_available")
        revision = backend.get("immutable_model_revision")
        if route == "precomputed-transcript":
            return (
                immutable is True
                and isinstance(revision, str)
                and bool(revision.strip())
                and revision == (self.job_config.get("source") or {}).get(
                    "transcript_sha256"
                )
            )
        if immutable is False:
            # A completed transcript is still a valid whole-stage checkpoint
            # for this immutable job/source snapshot. Unknown model identity
            # only disables lower-level cross-call model-output reuse.
            return revision is None
        if (
            immutable is not True
            or not isinstance(revision, str)
            or not revision.strip()
        ):
            return False
        asr_config = self.job_config.get("asr") or {}
        model_config = (
            asr_config.get("parakeet")
            if route in parakeet_routes
            else asr_config.get("whisper")
        ) or {}
        return model_config.get("model_revision") == revision

    def save_fixed(
        self,
        stage: str,
        expected_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        if self.source_identity is None:
            raise CheckpointError("Bind the final source before saving checkpoints")
        envelope = self._envelope(stage, expected_key, payload, provenance)
        if stage == "asr":
            route = envelope.get("route_context")
            if not isinstance(route, str) or not self._valid_asr_provenance(
                envelope, route
            ):
                return False
        if stage == "clip-plan":
            route = envelope.get("route_context")
            if not isinstance(route, dict) or not self._valid_clip_plan_payload(
                payload, route
            ):
                return False
        return self._save(self._path(stage), envelope)

    def load_keyed(self, stage: str, expected_key: str) -> dict | None:
        if self.source_identity is None:
            raise CheckpointError("Bind the final source before loading checkpoints")
        envelope = self._read(self._path(stage, expected_key))
        if envelope is None or not self._validate(envelope, stage, expected_key):
            return None
        return envelope

    def save_keyed(
        self,
        stage: str,
        expected_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        if self.source_identity is None:
            raise CheckpointError("Bind the final source before saving checkpoints")
        return self._save(
            self._path(stage, expected_key),
            self._envelope(stage, expected_key, payload, provenance),
        )

    def _resolved_artifact_path(self, descriptor: dict) -> str | None:
        relative = descriptor.get("relative_path") or descriptor.get("filename")
        if not isinstance(relative, str) or not relative or os.path.isabs(relative):
            return None
        path = os.path.realpath(os.path.join(self.project_dir, relative))
        if os.path.commonpath([self.project_dir, path]) != self.project_dir:
            return None
        return path

    def validate_artifact(self, descriptor: dict) -> bool:
        try:
            path = self._resolved_artifact_path(descriptor)
            if path is None or not os.path.isfile(path) or os.path.islink(path):
                return False
            if os.path.getsize(path) != int(descriptor["size"]):
                return False
            if sha256_file(path) != descriptor["sha256"]:
                return False
            # A descriptor that records probe success requires a currently
            # playable media file as well as matching bytes.
            if descriptor.get("probe"):
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", path,
                    ],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    return False
                try:
                    duration = float(result.stdout.strip())
                except (TypeError, ValueError):
                    return False
                if not math.isfinite(duration) or duration <= 0:
                    return False
            return True
        except Exception:
            return False

    @classmethod
    def open_existing_project(
        cls,
        project_dir: str,
        *,
        current_enablement: bool,
    ):
        config = job_state.load_job_config(project_dir)
        state = job_state.load_job_state(project_dir)
        if config["job_id"] != state["job_id"]:
            raise CheckpointError("Job config/state identities differ")
        if config["configuration_revision"] != state["configuration_revision"]:
            raise CheckpointError("Job config/state revisions differ")
        store = cls(
            project_dir,
            config,
            enabled=(
                current_enablement
                and bool(config.get("checkpointing_enabled"))
            ),
            execution_generation=state["execution_generation"],
        )
        if not store.enabled:
            return store
        source = store.load_source(expected_source_request_key(config))
        if source is None:
            raise CheckpointError("Project has no valid retained-source checkpoint")
        identity = source.get("source_fingerprint")
        store.bind_source(identity)
        return store

    def load_current_clip_plan(self, expected_key_for_route) -> dict | None:
        def validate(value):
            if not isinstance(value, dict):
                raise CheckpointError("clip-plan route is not an object")
            if value.get("selection_route") not in SELECTION_ROUTES:
                raise CheckpointError("unknown selection route")
            if value.get("asr_effective_route") not in ASR_ROUTES:
                raise CheckpointError("unknown ASR route")
            if value.get("asr_outcome") not in ASR_OUTCOMES:
                raise CheckpointError("unknown ASR outcome")
            revision = value.get("selection_model_revision")
            if revision is not None and (
                not isinstance(revision, str)
                or not revision
                or len(revision) > 256
                or not re.fullmatch(r"[A-Za-z0-9:._+-]+", revision)
            ):
                raise CheckpointError("invalid selection model revision")
            return dict(value)

        envelope = self.load_routed_fixed(
            "clip-plan", validate, expected_key_for_route,
        )
        if envelope is None or not self._valid_clip_plan_payload(
            envelope.get("payload"), envelope.get("route_context")
        ):
            return None
        return envelope

    def _valid_clip_plan_payload(self, payload: dict, route: dict) -> bool:
        if not isinstance(payload, dict):
            return False
        shorts = payload.get("shorts")
        if not isinstance(shorts, list) or not shorts or len(shorts) > 100:
            return False
        source_duration = float(
            (self.source_identity or {}).get("duration_ms", 0)
        ) / 1000.0
        if not math.isfinite(source_duration) or source_duration <= 0:
            return False
        seen_clips = set()
        selection = self.job_config.get("selection") or {}

        def configured_number(name, default):
            raw = selection.get(name)
            if raw is None or raw == "":
                return default
            return _finite_number(raw)

        minimum = configured_number("clip_min_seconds", 15.0)
        maximum = configured_number("clip_max_seconds", 60.0)
        if minimum is None or maximum is None:
            return False
        minimum = min(max(minimum, 5.0), 175.0)
        maximum = min(max(maximum, 10.0), 180.0)
        if maximum < minimum + 5.0:
            maximum = min(180.0, minimum + 5.0)
        explicit_ceiling = configured_number("target_max", None)
        if explicit_ceiling is not None and (
            explicit_ceiling < 1 or len(shorts) > int(explicit_ceiling)
        ):
            return False

        for clip in shorts:
            if not _valid_clip_candidate(
                clip,
                source_identity=self.source_identity,
            ):
                return False
            try:
                start = float(clip["start"])
                end = float(clip["end"])
            except (KeyError, TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in (start, end)):
                return False
            if start < 0 or end <= start or end > source_duration + 0.5:
                return False
            clip_duration = end - start
            if source_duration >= minimum and not (
                minimum <= clip_duration <= maximum
            ):
                return False
            clip_identity = (round(start, 6), round(end, 6))
            if clip_identity in seen_clips:
                return False
            seen_clips.add(clip_identity)
            for field in (
                "viral_hook_text",
                "video_title_for_youtube_short",
                "video_description_for_youtube",
            ):
                if field in clip and not isinstance(clip[field], str):
                    return False
        transcript = payload.get("transcript")
        if not _valid_embedded_source_analysis(payload.get("source_analysis")):
            return False
        speech_route = route.get("selection_route") in {
            "speech-local", "speech-gemini",
        }
        outcome = route.get("asr_outcome")
        if speech_route != (outcome == "speech"):
            return False
        if route.get("asr_effective_route") == "no-audio" \
                and outcome != "no_audio":
            return False
        if not _valid_transcript(
            transcript,
            source_duration=source_duration,
            require_segments=speech_route,
        ):
            return False
        return True

    def valid_render_artifacts(self, clip_plan_envelope: dict) -> dict[int, dict]:
        payload = clip_plan_envelope.get("payload") or {}
        shorts = payload.get("shorts") or []
        analysis = payload.get("source_analysis") or {}
        valid = {}
        for index, clip in enumerate(shorts):
            key = expected_render_key(
                self.job_config,
                self.source_identity,
                clip_plan_envelope,
                clip,
                analysis,
            )
            envelope = self.load_keyed("render", key)
            if envelope is None:
                continue
            render_payload = envelope.get("payload") or {}
            artifact = render_payload.get("artifact")
            expected_clip_identity = sha256_bytes(canonical_json_bytes(clip))
            if (
                render_payload.get("display_index") == index
                and render_payload.get("clip_identity") == expected_clip_identity
                and isinstance(artifact, dict)
                and artifact_matches_clip_duration(artifact, clip)
                and self.validate_artifact(artifact)
            ):
                valid[index] = envelope
        return valid


def completed_selection_boundary(
    config: dict,
    *,
    current_enablement: bool,
) -> str | None:
    """Return the furthest trustworthy selection boundary for recovery."""
    if not current_enablement or not config.get("checkpointing_enabled"):
        return None
    project_dir = config.get("project_dir")
    if not isinstance(project_dir, str) or not project_dir:
        return None
    try:
        store = CheckpointStore.open_existing_project(
            project_dir,
            current_enablement=current_enablement,
        )
        if store.load_current_clip_plan(
            lambda route: expected_clip_plan_key(
                config, store.source_identity, route,
            )
        ) is not None:
            return "complete"
        asr = store.load_routed_fixed(
            "asr",
            validate_asr_route,
            lambda route: expected_asr_key(
                config, store.source_identity, route,
            ),
        )
        outcome = (asr or {}).get("payload", {}).get("outcome")
        if outcome == "speech":
            return "speech"
        if outcome in {"no_audio", "sparse_speech"}:
            return "silent"
    except Exception:
        pass
    return None
