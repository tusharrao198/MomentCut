"""Crash-safe durable state and exclusive ownership for OpenShorts jobs.

This module deliberately contains no model/provider logic.  Critical job
state is written with a same-directory write/fsync/replace sequence and a
project lease prevents concurrent writers from publishing inconsistent state.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable


class JobStateError(RuntimeError):
    pass


class JobLeaseBusy(JobStateError):
    pass


class StateFileMissing(JobStateError):
    pass


class StateFileInvalid(JobStateError):
    pass


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ACTION_STATES = {
    "pending", "running", "dispatching", "done", "not_required",
    "permanent_failure", "manual_review",
}
_ACTION_ORDER = (
    "reservation",
    "archive",
    "autopilot_finalize",
    "autopilot_email",
    "autopilot_publish",
    "webhook",
    "alert",
    "proxy_usage",
    "clips_ready_email",
    "clip_activity",
)
_JOB_STATUSES = {
    "accepted", "queued", "processing", "waiting_for_credentials",
    "waiting_for_source", "completed", "partial", "failed",
}
_TERMINAL_STATUSES = {"completed", "partial", "failed"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_DURABLE_KEYS = {
    "api_key", "authorization", "cookie", "cookies", "password", "secret",
    "token", "webhook_secret",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_json(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("JSON state cannot contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalise_json(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            key = unicodedata.normalize("NFC", key)
            if key in result:
                raise ValueError("JSON object has duplicate keys after NFC normalization")
            result[key] = _normalise_json(item)
        return result
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _normalise_json(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_parent_directory(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(parent, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            # Windows and some bind mounts do not support directory fsync.
            if error.errno not in (
                errno.EBADF, errno.EINVAL, errno.ENOTSUP, errno.EACCES,
            ):
                raise
    finally:
        os.close(fd)


def _assert_not_symlink(path: str) -> None:
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise JobStateError(f"Refusing to replace symlink state path: {path}")
    except FileNotFoundError:
        pass


def atomic_write_bytes(path: str, payload: bytes, mode: int = 0o600) -> None:
    """Atomically publish bytes after flushing the file to stable storage."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise JobStateError(f"State directory does not exist: {parent}")
    _assert_not_symlink(path)
    name = os.path.basename(path)
    temporary = os.path.join(parent, f".{name}.{uuid.uuid4().hex}.tmp")
    fd = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb", closefd=True) as output:
            fd = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        for attempt in range(3):
            try:
                _assert_not_symlink(path)
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))
        try:
            os.chmod(path, mode)
        except OSError:
            pass
        _fsync_parent_directory(path)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: str, payload: dict, mode: int = 0o600) -> None:
    if not isinstance(payload, dict):
        raise TypeError("state JSON root must be an object")
    atomic_write_bytes(path, _json_bytes(payload), mode=mode)


def read_json(path: str, *, max_bytes: int | None = None) -> dict | None:
    try:
        with open(path, "rb") as source:
            if max_bytes is not None and os.fstat(source.fileno()).st_size > max_bytes:
                return None
            value = json.load(source)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_required_json(path: str, validator: Callable[[dict], Any]) -> dict:
    try:
        with open(path, "rb") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise StateFileMissing(f"Required state file is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateFileInvalid(f"Required state file is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise StateFileInvalid(f"Required state file is not a JSON object: {path}")
    try:
        result = validator(value)
    except StateFileInvalid:
        raise
    except Exception as error:
        raise StateFileInvalid(f"Required state file failed validation: {path}") from error
    if result is False:
        raise StateFileInvalid(f"Required state file failed validation: {path}")
    return value


def _validated_project(project_dir: str) -> str:
    path = os.path.realpath(os.path.abspath(os.fspath(project_dir)))
    if not os.path.isdir(path):
        raise JobStateError(f"Project directory does not exist: {project_dir}")
    return path


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise JobStateError("job_id contains unsupported characters")
    return job_id


def stable_job_paths(project_dir: str, job_id: str) -> dict:
    project = _validated_project(project_dir)
    job_id = _validate_job_id(job_id)
    return {
        "project_dir": project,
        "job_config": os.path.join(project, ".job-config.json"),
        "job_state": os.path.join(project, ".job-state.json"),
        "resume_manifest": os.path.join(project, ".resume.json"),
        "resume_secrets": os.path.join(project, ".resume-secrets.enc"),
        "run_lock": os.path.join(project, ".run.lock"),
        "source": os.path.join(project, "source.mp4"),
        "metadata": os.path.join(project, f"{job_id}_metadata.json"),
        "checkpoints": os.path.join(project, ".checkpoints"),
    }


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    for pure in (PurePosixPath(value), PureWindowsPath(value)):
        if pure.is_absolute() or pure.drive or ".." in pure.parts:
            return False
    if os.path.isabs(value):
        return False
    normalized = os.path.normpath(value)
    return normalized not in {".", ".."} and not normalized.startswith(
        f"..{os.sep}"
    )


def _contains_forbidden_durable_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _FORBIDDEN_DURABLE_KEYS or lowered.endswith(
                ("_password", "_secret", "_token", "_api_key")
            ):
                return True
            if _contains_forbidden_durable_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_durable_key(item) for item in value)
    return False


def _valid_model_descriptor(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    available = value.get("immutable_revision_available")
    revision = value.get("immutable_model_revision")
    if not isinstance(available, bool):
        return False
    if available is True and (
        not isinstance(revision, str) or not revision.strip() or len(revision) > 512
    ):
        return False
    if available is False and revision is not None:
        return False
    endpoint = value.get("endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, str) or len(endpoint) > 2048:
            return False
        try:
            from urllib.parse import urlsplit
            parsed = urlsplit(endpoint)
        except ValueError:
            return False
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            return False
    return True


def _validate_job_config(config: dict) -> bool:
    stored_hash = config.get("job_config_hash")
    unhashed = dict(config)
    unhashed.pop("job_config_hash", None)
    source = config.get("source")
    retained = source.get("retained_relative_path") if isinstance(source, dict) else None
    transcript = source.get("transcript_relative_path") if isinstance(source, dict) else None
    transcript_sha256 = source.get("transcript_sha256") \
        if isinstance(source, dict) else None
    project_dir = config.get("project_dir")
    environment = config.get("environment", {})
    requirements = config.get("credential_requirements", {})
    return (
        config.get("schema_version") == 1
        and isinstance(config.get("job_id"), str)
        and bool(_JOB_ID_RE.fullmatch(config["job_id"]))
        and isinstance(config.get("configuration_revision"), int)
        and config["configuration_revision"] >= 1
        and isinstance(source, dict)
        and source.get("kind") in {
            "upload", "url", "thumbnail_session", "upload_slot",
        }
        and _safe_relative_path(retained)
        and (transcript is None or _safe_relative_path(transcript))
        and (
            transcript_sha256 is None
            or isinstance(transcript_sha256, str)
            and bool(_SHA256_RE.fullmatch(transcript_sha256))
        )
        and isinstance(config.get("pipeline"), dict)
        and isinstance(environment, dict)
        and all(
            isinstance(name, str)
            and bool(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name))
            and isinstance(value, str)
            and len(value) <= 4096
            for name, value in environment.items()
        )
        and isinstance(requirements, dict)
        and set(requirements).issubset({
            "gemini", "gemini_speech", "gemini_silent", "llm_authentication",
            "source_authorization", "private_source",
        })
        and all(isinstance(value, bool) for value in requirements.values())
        and (
            "checkpointing_enabled" not in config
            or isinstance(config.get("checkpointing_enabled"), bool)
        )
        and (
            project_dir is None
            or isinstance(project_dir, str) and os.path.isabs(project_dir)
        )
        and (
            "text_selection" not in config
            or _valid_model_descriptor(config.get("text_selection"))
        )
        and (
            "vision_selection" not in config
            or _valid_model_descriptor(config.get("vision_selection"))
        )
        and not _contains_forbidden_durable_key(unhashed)
        and isinstance(stored_hash, str)
        and bool(_SHA256_RE.fullmatch(stored_hash))
        and stored_hash == hashlib.sha256(_json_bytes(unhashed)).hexdigest()
    )


def _valid_terminal_artifact(descriptor: Any, expected_index: int) -> bool:
    if not isinstance(descriptor, dict):
        return False
    if descriptor.get("clip_index") != expected_index:
        return False
    if not _safe_relative_path(descriptor.get("relative_path")):
        return False
    if (
        not isinstance(descriptor.get("size"), int)
        or descriptor["size"] <= 0
        or not isinstance(descriptor.get("duration_ms"), int)
        or descriptor["duration_ms"] <= 0
        or not isinstance(descriptor.get("sha256"), str)
        or not _SHA256_RE.fullmatch(descriptor["sha256"])
        or not isinstance(descriptor.get("clip_plan_identity"), str)
        or not _SHA256_RE.fullmatch(descriptor["clip_plan_identity"])
        or descriptor.get("probe") != {"playable": True}
    ):
        return False
    companions = descriptor.get("companions", [])
    if not isinstance(companions, list):
        return False
    seen = set()
    for companion in companions:
        if (
            not isinstance(companion, dict)
            or not _safe_relative_path(companion.get("relative_path"))
            or companion.get("relative_path") in seen
            or not isinstance(companion.get("size"), int)
            or companion["size"] <= 0
            or not isinstance(companion.get("sha256"), str)
            or not _SHA256_RE.fullmatch(companion["sha256"])
            or not isinstance(companion.get("duration_ms"), int)
            or companion["duration_ms"] <= 0
            or companion.get("probe") != {"playable": True}
        ):
            return False
        seen.add(companion["relative_path"])
    return True


def _autopilot_plan_id(payload: dict) -> str:
    """Match cloud.autopilot's deterministic plan identity without importing it."""
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _valid_autopilot_item(item: Any, parent_event_id: str) -> bool:
    if not isinstance(item, dict):
        return False
    identity = {
        "parent_event_id": item.get("parent_event_id"),
        "clip_key": item.get("clip_key"),
        "clip_index": item.get("clip_index"),
        "profile_id": item.get("profile_id"),
        "platforms": item.get("platforms"),
        "scheduled_at": item.get("scheduled_at"),
    }
    clip = item.get("clip")
    allowed_clip_fields = {
        "title",
        "video_title_for_youtube_short",
        "video_description_for_youtube",
        "video_description_for_instagram",
        "video_description_for_tiktok",
    }
    if (
        identity["parent_event_id"] != parent_event_id
        or not isinstance(identity["clip_key"], str)
        or not identity["clip_key"]
        or len(identity["clip_key"]) > 512
        or not isinstance(identity["clip_index"], int)
        or isinstance(identity["clip_index"], bool)
        or identity["clip_index"] < 0
        or not isinstance(identity["profile_id"], str)
        or not identity["profile_id"]
        or len(identity["profile_id"]) > 256
        or not isinstance(identity["platforms"], list)
        or not identity["platforms"]
        or not all(
            isinstance(platform, str)
            and bool(re.fullmatch(r"[a-z0-9_-]{1,32}", platform))
            for platform in identity["platforms"]
        )
        or not isinstance(identity["scheduled_at"], str)
        or not identity["scheduled_at"]
        or len(identity["scheduled_at"]) > 64
        or not _safe_relative_path(item.get("relative_path"))
        or not isinstance(item.get("run_id"), str)
        or not item["run_id"]
        or len(item["run_id"]) > 256
        or not isinstance(item.get("timezone"), str)
        or not item["timezone"]
        or len(item["timezone"]) > 128
        or not isinstance(clip, dict)
        or set(clip) - allowed_clip_fields
        or any(
            value is not None and not isinstance(value, str)
            for value in clip.values()
        )
    ):
        return False
    item_id = _autopilot_plan_id(identity)
    event_id = _autopilot_plan_id({
        "namespace": "openshorts-autopilot-publish-v1",
        "item_id": item_id,
    })
    return item.get("item_id") == item_id and item.get("event_id") == event_id


def _valid_terminal_actions(
    actions: Any,
    job_id: str,
    terminal_revision: int,
) -> bool:
    if not isinstance(actions, dict) or set(actions) != set(_ACTION_ORDER):
        return False
    for name, record in actions.items():
        if (
            not isinstance(record, dict)
            or record.get("state") not in _ACTION_STATES
            or not isinstance(record.get("attempts"), int)
            or record["attempts"] < 0
            or not isinstance(record.get("event_id"), str)
            or not _SHA256_RE.fullmatch(record["event_id"])
            or record["event_id"] != _event_id(
                job_id, terminal_revision, name,
            )
        ):
            return False
        if name == "autopilot_publish":
            items = record.get("items", [])
            if not isinstance(items, list):
                return False
            item_ids = set()
            for item in items:
                if (
                    not _valid_autopilot_item(item, record["event_id"])
                    or item.get("state") not in _ACTION_STATES
                    or not isinstance(item.get("attempts"), int)
                    or isinstance(item.get("attempts"), bool)
                    or item["attempts"] < 0
                    or item["item_id"] in item_ids
                ):
                    return False
                item_ids.add(item["item_id"])
    return True


_TERMINAL_CONTEXT_KEYS = {
    "job_id",
    "terminal_revision",
    "reservation_id",
    "user_id",
    "webhook_url",
    "webhook_secret_required",
    "base_url",
    "api_caller",
    "watermark",
    "user_plan",
    "proxy_route",
    "proxy_bytes",
    "autopilot_run_id",
}


def _valid_terminal_context(
    context: Any,
    job_id: str,
    terminal_revision: int,
) -> bool:
    if (
        not isinstance(context, dict)
        or set(context) - _TERMINAL_CONTEXT_KEYS
        or context.get("job_id") != job_id
        or context.get("terminal_revision") != terminal_revision
    ):
        return False
    for name in (
        "reservation_id", "user_id", "webhook_url", "base_url",
        "user_plan", "autopilot_run_id",
    ):
        value = context.get(name)
        if value is not None and (
            not isinstance(value, str) or len(value) > 4096
        ):
            return False
    proxy_route = context.get("proxy_route")
    if proxy_route is not None:
        if not isinstance(proxy_route, dict):
            return False
        try:
            if len(_json_bytes(proxy_route)) > 65536:
                return False
        except (TypeError, ValueError):
            return False
    for name in ("webhook_secret_required", "api_caller", "watermark"):
        if name in context and not isinstance(context.get(name), bool):
            return False
    proxy_bytes = context.get("proxy_bytes")
    return (
        proxy_bytes is None
        or isinstance(proxy_bytes, int)
        and not isinstance(proxy_bytes, bool)
        and proxy_bytes >= 0
    )


def _validate_job_state(state: dict) -> bool:
    basic = (
        state.get("schema_version") == 1
        and isinstance(state.get("job_id"), str)
        and bool(_JOB_ID_RE.fullmatch(state["job_id"]))
        and isinstance(state.get("configuration_revision"), int)
        and isinstance(state.get("execution_generation"), int)
        and state["configuration_revision"] >= 1
        and state["execution_generation"] >= 0
        and state.get("status") in _JOB_STATUSES
        and not _contains_forbidden_durable_key(state)
        and (
            state.get("owner_id") is None
            or isinstance(state.get("owner_id"), str)
        )
        and (
            "retry_count" not in state
            or isinstance(state.get("retry_count"), int)
            and state["retry_count"] >= 0
        )
    )
    if not basic:
        return False
    live_plan = state.get("live_plan")
    if live_plan is not None and not (
        isinstance(live_plan, dict)
        and isinstance(live_plan.get("execution_generation"), int)
        and live_plan["execution_generation"] == state["execution_generation"]
        and _safe_relative_path(live_plan.get("metadata_filename"))
        and isinstance(live_plan.get("plan_hash"), str)
        and bool(_SHA256_RE.fullmatch(live_plan["plan_hash"]))
        and isinstance(live_plan.get("count"), int)
        and live_plan["count"] > 0
    ):
        return False
    if state["status"] not in _TERMINAL_STATUSES:
        return (
            "terminal_revision" not in state
            and "terminal_artifacts" not in state
            and "terminal_actions" not in state
            and "terminal_action_context" not in state
        )
    artifacts = state.get("terminal_artifacts")
    missing = state.get("missing_clip_indices")
    if (
        not isinstance(state.get("terminal_revision"), int)
        or state["terminal_revision"] < 1
        or not isinstance(artifacts, dict)
        or not isinstance(missing, list)
        or any(not isinstance(index, int) or index < 0 for index in missing)
        or len(set(missing)) != len(missing)
        or not _valid_terminal_context(
            state.get("terminal_action_context"),
            state["job_id"],
            state["terminal_revision"],
        )
        or not _valid_terminal_actions(
            state.get("terminal_actions"),
            state["job_id"],
            state["terminal_revision"],
        )
    ):
        return False
    artifact_indices = set()
    artifact_paths = set()
    for raw_index, descriptor in artifacts.items():
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            return False
        index = int(raw_index)
        if index in artifact_indices or not _valid_terminal_artifact(descriptor, index):
            return False
        paths = [
            descriptor["relative_path"],
            *[
                companion["relative_path"]
                for companion in descriptor.get("companions", [])
            ],
        ]
        if len(set(paths)) != len(paths) or artifact_paths.intersection(paths):
            return False
        artifact_paths.update(paths)
        artifact_indices.add(index)
    if state["status"] == "completed" and (missing or not artifacts):
        return False
    if state["status"] == "partial" and (not missing or not artifacts):
        return False
    if state["status"] == "failed" and artifacts:
        return False
    return not artifact_indices.intersection(missing)


def _validate_resume_manifest(manifest: dict) -> bool:
    heartbeat = manifest.get("heartbeat")
    priority = manifest.get("priority", 2)
    partial = manifest.get("partial")
    source_cap = manifest.get("source_cap_minutes")
    return (
        manifest.get("schema_version", 1) == 1
        and isinstance(manifest.get("job_id"), str)
        and bool(_JOB_ID_RE.fullmatch(manifest["job_id"]))
        and isinstance(manifest.get("configuration_revision"), int)
        and manifest["configuration_revision"] >= 1
        and isinstance(manifest.get("attempts", 0), int)
        and not isinstance(manifest.get("attempts", 0), bool)
        and manifest.get("attempts", 0) >= 0
        and isinstance(priority, int)
        and not isinstance(priority, bool)
        and 0 <= priority <= 100
        and (
            heartbeat is None
            or isinstance(heartbeat, (int, float))
            and not isinstance(heartbeat, bool)
            and float("-inf") < float(heartbeat) < float("inf")
            and heartbeat >= 0
        )
        and (
            manifest.get("instance") is None
            or isinstance(manifest.get("instance"), str)
            and len(manifest["instance"]) <= 256
        )
        and (
            manifest.get("user_id") is None
            or isinstance(manifest.get("user_id"), str)
            and len(manifest["user_id"]) <= 256
        )
        and (
            manifest.get("reservation_id") is None
            or isinstance(manifest.get("reservation_id"), str)
            and len(manifest["reservation_id"]) <= 256
        )
        and (
            manifest.get("webhook_url") is None
            or isinstance(manifest.get("webhook_url"), str)
            and len(manifest["webhook_url"]) <= 4096
        )
        and manifest.get("resume_secret_purpose") in {None, "job-runtime"}
        and (
            manifest.get("base_url") is None
            or isinstance(manifest.get("base_url"), str)
            and len(manifest["base_url"]) <= 4096
        )
        and (
            partial is None
            or isinstance(partial, dict)
            and set(partial) == {"processed_minutes", "total_minutes"}
            and all(
                isinstance(partial.get(name), (int, float))
                and not isinstance(partial.get(name), bool)
                and float(partial[name]) > 0
                for name in ("processed_minutes", "total_minutes")
            )
            and float(partial["processed_minutes"])
            <= float(partial["total_minutes"])
        )
        and (
            source_cap is None
            or isinstance(source_cap, (int, float))
            and not isinstance(source_cap, bool)
            and float(source_cap) > 0
        )
        and all(
            name not in manifest or isinstance(manifest.get(name), bool)
            for name in ("watermark", "skip_statics", "api_caller")
        )
        and "cmd" not in manifest
        and "webhook_secret" not in manifest
        and not _contains_forbidden_durable_key(manifest)
    )


def seal_job_config(config: dict) -> dict:
    """Return a copy carrying the canonical integrity hash used on disk."""
    if not isinstance(config, dict):
        raise StateFileInvalid("Job configuration must be an object")
    sealed = dict(config)
    sealed.pop("job_config_hash", None)
    sealed["job_config_hash"] = hashlib.sha256(_json_bytes(sealed)).hexdigest()
    return sealed


def write_job_config(project_dir: str, config: dict) -> None:
    project = _validated_project(project_dir)
    if not _validate_job_config(config):
        raise StateFileInvalid("Refusing to write invalid job configuration")
    configured_project = config.get("project_dir")
    if configured_project is not None and os.path.realpath(
        os.path.abspath(os.fspath(configured_project))
    ) != project:
        raise StateFileInvalid("Job configuration is bound to another project directory")
    path = os.path.join(project, ".job-config.json")
    if os.path.exists(path):
        existing = read_required_json(path, _validate_job_config)
        if existing != config:
            raise StateFileInvalid(
                "Immutable job configuration already exists and differs"
            )
        return
    atomic_write_json(path, config)


def load_job_config(project_dir: str) -> dict:
    project = _validated_project(project_dir)
    config = read_required_json(
        os.path.join(project, ".job-config.json"), _validate_job_config,
    )
    configured_project = config.get("project_dir")
    if configured_project is not None and os.path.realpath(
        os.path.abspath(os.fspath(configured_project))
    ) != project:
        raise StateFileInvalid("Job configuration is bound to another project directory")
    return config


def write_job_state(project_dir: str, state: dict) -> None:
    if not _validate_job_state(state):
        raise StateFileInvalid("Refusing to write invalid job state")
    atomic_write_json(os.path.join(project_dir, ".job-state.json"), state)


def load_job_state(project_dir: str) -> dict:
    return read_required_json(
        os.path.join(project_dir, ".job-state.json"), _validate_job_state,
    )


def write_resume_manifest(project_dir: str, manifest: dict) -> None:
    if not _validate_resume_manifest(manifest):
        raise StateFileInvalid("Refusing to write unsafe resume manifest")
    atomic_write_json(os.path.join(project_dir, ".resume.json"), manifest)


def load_resume_manifest(project_dir: str) -> dict | None:
    path = os.path.join(project_dir, ".resume.json")
    if not os.path.exists(path):
        return None
    return read_required_json(path, _validate_resume_manifest)


def _decode_key(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        key = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as error:
        raise JobStateError("RESUME_SECRET_KEYS contains invalid base64url") from error
    if len(key) != 32:
        raise JobStateError("Every resume-secret key must decode to exactly 32 bytes")
    return key


def load_resume_secret_keyring(env: dict | None = None) -> dict:
    source = os.environ if env is None else env
    active = str(source.get("RESUME_SECRET_ACTIVE_KEY_ID", "")).strip()
    raw_keys = str(source.get("RESUME_SECRET_KEYS", "")).strip()
    if not active and not raw_keys:
        return {"active_key_id": None, "keys": {}}
    if not active or not raw_keys:
        raise JobStateError(
            "RESUME_SECRET_ACTIVE_KEY_ID and RESUME_SECRET_KEYS must be set together"
        )
    keys = {}
    for entry in raw_keys.split(","):
        try:
            key_id, encoded = entry.strip().split(":", 1)
        except ValueError as error:
            raise JobStateError("RESUME_SECRET_KEYS entries must be key_id:base64url") from error
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key_id) or key_id in keys:
            raise JobStateError("Resume-secret key IDs must be unique safe identifiers")
        keys[key_id] = _decode_key(encoded.strip())
    if active not in keys:
        raise JobStateError("Active resume-secret key ID is not present in keyring")
    return {"active_key_id": active, "keys": keys}


def _secret_aad(job_id: str, configuration_revision: int, purpose: str) -> bytes:
    return _json_bytes({
        "job_id": job_id,
        "configuration_revision": configuration_revision,
        "purpose": purpose,
        "format_version": 1,
    })


def write_resume_secrets(
    project_dir: str,
    job_id: str,
    configuration_revision: int,
    purpose: str,
    secrets: dict,
    keyring: dict,
) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(secrets, dict) or not secrets:
        raise JobStateError("Resume-secret payload must be a non-empty object")
    active = keyring.get("active_key_id")
    key = (keyring.get("keys") or {}).get(active)
    if not active or not key:
        raise JobStateError("No active resume-secret encryption key is configured")
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(
        nonce, _json_bytes(secrets),
        _secret_aad(job_id, configuration_revision, purpose),
    )
    envelope = {
        "format_version": 1,
        "key_id": active,
        "nonce": base64.urlsafe_b64encode(nonce).decode("ascii").rstrip("="),
        "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii").rstrip("="),
    }
    atomic_write_json(os.path.join(project_dir, ".resume-secrets.enc"), envelope)


def load_resume_secrets(
    project_dir: str,
    job_id: str,
    configuration_revision: int,
    purpose: str,
    keyring: dict,
) -> dict | None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    path = os.path.join(project_dir, ".resume-secrets.enc")
    if not os.path.exists(path):
        return None
    envelope = read_required_json(
        path,
        lambda value: (
            value.get("format_version") == 1
            and isinstance(value.get("key_id"), str)
            and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value["key_id"]))
            and isinstance(value.get("nonce"), str)
            and isinstance(value.get("ciphertext"), str)
        ),
    )
    key_id = envelope.get("key_id")
    key = (keyring.get("keys") or {}).get(key_id)
    if key is None:
        raise StateFileInvalid(f"Resume-secret decryption key is unavailable: {key_id}")
    try:
        nonce = base64.urlsafe_b64decode(
            str(envelope["nonce"]) + "=" * (-len(str(envelope["nonce"])) % 4)
        )
        ciphertext = base64.urlsafe_b64decode(
            str(envelope["ciphertext"])
            + "=" * (-len(str(envelope["ciphertext"])) % 4)
        )
        plaintext = AESGCM(key).decrypt(
            nonce, ciphertext,
            _secret_aad(job_id, configuration_revision, purpose),
        )
        value = json.loads(plaintext)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, InvalidTag) as error:
        raise StateFileInvalid("Resume-secret authentication failed") from error
    if not isinstance(value, dict):
        raise StateFileInvalid("Resume-secret plaintext is not an object")
    return value


def _event_id(job_id: str, terminal_revision: int, action: str) -> str:
    return hashlib.sha256(_json_bytes({
        "namespace": "openshorts-terminal-v1",
        "job_id": job_id,
        "terminal_revision": terminal_revision,
        "action": action,
    })).hexdigest()


def required_terminal_actions(
    terminal_status: str,
    action_context: dict,
) -> dict:
    if terminal_status not in {"completed", "partial", "failed"}:
        raise JobStateError(f"Unsupported terminal status: {terminal_status}")
    if not isinstance(action_context, dict):
        raise JobStateError("action_context must be an object")
    job_id = action_context.get("job_id")
    terminal_revision = action_context.get("terminal_revision")
    if not isinstance(job_id, str) or not isinstance(terminal_revision, int):
        raise JobStateError("Terminal action context lacks identity/revision")
    enabled = action_context.get("actions") or {}
    if (
        not isinstance(enabled, dict)
        or set(enabled) - set(_ACTION_ORDER)
        or any(not isinstance(value, bool) for value in enabled.values())
    ):
        raise JobStateError("Terminal action selection is invalid")
    actions = {}
    for name in _ACTION_ORDER:
        required = bool(enabled.get(name, False))
        record = {
            "state": "pending" if required else "not_required",
            "attempts": 0,
            "event_id": _event_id(job_id, terminal_revision, name),
        }
        if name == "autopilot_publish":
            record["items"] = list(action_context.get("autopilot_publish_items") or [])
            if required and not record["items"]:
                record["state"] = "not_required"
        actions[name] = record
    return actions


def commit_terminal_state(
    project_dir: str,
    expected_execution_generation: int,
    terminal_status: str,
    terminal_artifacts: dict,
    missing_clip_indices: list[int],
    action_context: dict,
) -> dict:
    state = load_job_state(project_dir)
    if state["execution_generation"] != expected_execution_generation:
        raise JobStateError("Execution generation changed before terminal commit")
    if state.get("status") in {"completed", "partial", "failed"}:
        return state
    if terminal_status not in {"completed", "partial", "failed"}:
        raise JobStateError("Invalid terminal status")
    if not isinstance(terminal_artifacts, dict):
        raise JobStateError("terminal_artifacts must be an object")
    missing = []
    for index in missing_clip_indices:
        if not isinstance(index, int) or index < 0:
            raise JobStateError("missing clip indices must be non-negative integers")
        if index not in missing:
            missing.append(index)
    if terminal_status == "completed" and (missing or not terminal_artifacts):
        raise JobStateError("Completed terminal state requires all artifacts")
    if terminal_status == "partial" and (not missing or not terminal_artifacts):
        raise JobStateError("Partial terminal state requires artifacts and misses")
    if terminal_status == "failed" and terminal_artifacts:
        raise JobStateError("Failed terminal state cannot deliver artifacts")
    revision = int(state.get("terminal_revision") or 0) + 1
    context = dict(action_context or {})
    context["job_id"] = state["job_id"]
    context["terminal_revision"] = revision
    new_state = dict(state)
    new_state.update({
        "status": terminal_status,
        "terminal_revision": revision,
        "terminal_at": _utc_now(),
        "terminal_artifacts": terminal_artifacts,
        "missing_clip_indices": sorted(missing),
        # Durable non-secret inputs let another process finish the outbox
        # without reconstructing intent from volatile in-memory job fields.
        "terminal_action_context": {
            key: value
            for key, value in context.items()
            if key not in {"actions", "autopilot_publish_items"}
        },
        "terminal_actions": required_terminal_actions(terminal_status, context),
    })
    write_job_state(project_dir, new_state)
    return new_state


def transition_terminal_action(
    project_dir: str,
    action: str,
    expected_state: str,
    new_state: str,
    details: dict | None = None,
) -> dict:
    if new_state not in _ACTION_STATES:
        raise JobStateError(f"Invalid terminal action state: {new_state}")
    state = load_job_state(project_dir)
    actions = dict(state.get("terminal_actions") or {})
    if action not in actions:
        raise JobStateError(f"Unknown terminal action: {action}")
    record = dict(actions[action])
    if record.get("state") != expected_state:
        raise JobStateError(
            f"Terminal action {action} is {record.get('state')}, expected {expected_state}"
        )
    record["state"] = new_state
    if new_state in {"running", "dispatching"}:
        record["attempts"] = int(record.get("attempts") or 0) + 1
    record["updated_at"] = _utc_now()
    if details:
        record.update(details)
    actions[action] = record
    state["terminal_actions"] = actions
    write_job_state(project_dir, state)
    return state


def transition_terminal_action_item(
    project_dir: str,
    action: str,
    item_id: str,
    expected_state: str,
    new_state: str,
    details: dict | None = None,
) -> dict:
    if new_state not in _ACTION_STATES:
        raise JobStateError(f"Invalid terminal action item state: {new_state}")
    state = load_job_state(project_dir)
    actions = dict(state.get("terminal_actions") or {})
    parent = dict(actions.get(action) or {})
    items = [dict(item) for item in parent.get("items") or []]
    matches = [item for item in items if item.get("item_id") == item_id]
    if len(matches) != 1:
        raise JobStateError(f"Unknown or duplicate terminal action item: {item_id}")
    item = matches[0]
    if item.get("state") != expected_state:
        raise JobStateError(
            f"Terminal action item {item_id} is {item.get('state')}, expected {expected_state}"
        )
    item["state"] = new_state
    if new_state in {"running", "dispatching"}:
        item["attempts"] = int(item.get("attempts") or 0) + 1
    item["updated_at"] = _utc_now()
    if details:
        item.update(details)
    item_states = {entry.get("state") for entry in items}
    if item_states <= {"done", "not_required"}:
        parent["state"] = "done"
    elif "manual_review" in item_states:
        parent["state"] = "manual_review"
    elif "permanent_failure" in item_states:
        parent["state"] = "permanent_failure"
    elif item_states & {"running", "dispatching"}:
        parent["state"] = "running"
    else:
        parent["state"] = "pending"
    parent["items"] = items
    actions[action] = parent
    state["terminal_actions"] = actions
    write_job_state(project_dir, state)
    return state


def configure_autopilot_actions(project_dir: str, plan: dict) -> dict:
    """Install the durable email/per-clip plan produced by Autopilot finalize."""
    if not isinstance(plan, dict):
        raise JobStateError("Autopilot action plan must be an object")
    state = load_job_state(project_dir)
    actions = dict(state.get("terminal_actions") or {})
    email = dict(actions.get("autopilot_email") or {})
    publish = dict(actions.get("autopilot_publish") or {})
    clips_ready = dict(actions.get("clips_ready_email") or {})
    if not email or not publish or not clips_ready:
        raise JobStateError("Terminal outbox lacks Autopilot actions")

    email_plan = plan.get("email")
    if isinstance(email_plan, dict):
        if email.get("state") == "not_required":
            email["state"] = "pending"
            email["attempts"] = 0
        email["plan"] = email_plan
    elif email.get("state") == "not_required":
        email["plan"] = None

    raw_items = plan.get("publish_items") or []
    if not isinstance(raw_items, list):
        raise JobStateError("Autopilot publish plan must contain a list")
    if raw_items and publish.get("state") == "not_required":
        items = []
        seen = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise JobStateError("Autopilot publish item must be an object")
            item_id = raw.get("item_id")
            event_id = raw.get("event_id")
            if (
                not isinstance(item_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", item_id)
                or item_id in seen
                or not isinstance(event_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", event_id)
            ):
                raise JobStateError("Autopilot publish item identity is invalid")
            if (
                _contains_forbidden_durable_key(raw)
                or not isinstance(raw.get("clip_key"), str)
                or not raw["clip_key"]
                or len(raw["clip_key"]) > 512
                or not isinstance(raw.get("clip_index"), int)
                or raw["clip_index"] < 0
                or not isinstance(raw.get("profile_id"), str)
                or not raw["profile_id"]
                or len(raw["profile_id"]) > 256
                or not isinstance(raw.get("platforms"), list)
                or not raw["platforms"]
                or not all(
                    isinstance(platform, str)
                    and re.fullmatch(r"[a-z0-9_-]{1,32}", platform)
                    for platform in raw["platforms"]
                )
                or not isinstance(raw.get("scheduled_at"), str)
                or len(raw["scheduled_at"]) > 64
                or not _safe_relative_path(raw.get("relative_path"))
            ):
                raise JobStateError("Autopilot publish item payload is invalid")
            seen.add(item_id)
            items.append({
                **raw,
                "state": "pending",
                "attempts": 0,
                "last_error_code": None,
            })
        publish["items"] = items
        publish["state"] = "pending"
    elif not raw_items and publish.get("state") == "not_required":
        publish["items"] = []

    actions["autopilot_email"] = email
    actions["autopilot_publish"] = publish
    if plan.get("is_autopilot") and clips_ready.get("state") == "pending":
        # The dedicated Autopilot email replaces the ordinary completion
        # message. Persist that choice so restart recovery cannot send both.
        clips_ready["state"] = "not_required"
        clips_ready["updated_at"] = _utc_now()
    actions["clips_ready_email"] = clips_ready
    state["terminal_actions"] = actions
    write_job_state(project_dir, state)
    return state


class JobLease:
    def __init__(self, path: str, handle):
        self.path = path
        self._handle = handle

    @classmethod
    def acquire(
        cls,
        project_dir: str,
        job_id: str,
        *,
        blocking: bool = False,
    ):
        paths = stable_job_paths(project_dir, job_id)
        handle = open(paths["run_lock"], "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if os.path.getsize(paths["run_lock"]) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                try:
                    msvcrt.locking(handle.fileno(), mode, 1)
                except OSError as error:
                    raise JobLeaseBusy(f"Job {job_id} is already running") from error
            else:
                import fcntl
                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(handle.fileno(), flags)
                except OSError as error:
                    if error.errno in (errno.EACCES, errno.EAGAIN):
                        raise JobLeaseBusy(f"Job {job_id} is already running") from error
                    raise
            return cls(paths["run_lock"], handle)
        except Exception:
            handle.close()
            raise

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
