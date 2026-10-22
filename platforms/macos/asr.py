"""Lazy Apple-Silicon ASR adapters.

Importing this module is safe on every platform: MLX and Hugging Face are
loaded only when an Apple-Silicon ASR route is actually invoked.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from accelerator_runtime import AcceleratorConfigurationError


class MacASRConfigurationError(AcceleratorConfigurationError):
    """The selected MLX model cannot be resolved deterministically."""


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_WHISPER_MODEL_ALIASES = {
    "small": (
        "mlx-community/whisper-small-mlx",
        "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
    ),
    "large-v3-turbo": (
        "mlx-community/whisper-large-v3-turbo",
        "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    ),
}
_PARAKEET_MODEL = (
    "mlx-community/parakeet-tdt-0.6b-v3",
    "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
)
_whisper_lock = threading.Lock()
_parakeet_lock = threading.Lock()
_parakeet_model = None
_parakeet_key = None
_mlx_import_error = None


def resolve_whisper_model():
    """Return the configured ``(repository, immutable revision)`` pair."""
    alias = os.environ.get("WHISPER_MODEL", "small").strip()
    repository = os.environ.get("WHISPER_MLX_MODEL", "").strip()
    revision = os.environ.get("WHISPER_MLX_MODEL_REVISION", "").strip().lower()

    default = _WHISPER_MODEL_ALIASES.get(alias)
    if not repository:
        if default is None:
            raise MacASRConfigurationError(
                f"WHISPER_MODEL={alias!r} has no tested MLX mapping; set "
                "WHISPER_MLX_MODEL and WHISPER_MLX_MODEL_REVISION"
            )
        repository, default_revision = default
        revision = revision or default_revision
    elif not revision and default and repository == default[0]:
        revision = default[1]

    if not repository:
        raise MacASRConfigurationError("WHISPER_MLX_MODEL cannot be empty")
    if not _REVISION_RE.fullmatch(revision):
        raise MacASRConfigurationError(
            "WHISPER_MLX_MODEL_REVISION must be a 40-character commit SHA"
        )
    return repository, revision


def resolve_parakeet_model():
    """Return the configured immutable Parakeet MLX model identity."""
    repository = os.environ.get(
        "PARAKEET_MLX_MODEL", _PARAKEET_MODEL[0]).strip()
    revision = os.environ.get("PARAKEET_MLX_MODEL_REVISION", "").strip().lower()
    if not revision and repository == _PARAKEET_MODEL[0]:
        revision = _PARAKEET_MODEL[1]
    if not repository:
        raise MacASRConfigurationError("PARAKEET_MLX_MODEL cannot be empty")
    if not _REVISION_RE.fullmatch(revision):
        raise MacASRConfigurationError(
            "PARAKEET_MLX_MODEL_REVISION must be a 40-character commit SHA"
        )
    return repository, revision


def _snapshot_path(repository, revision, *, local_files_only=False):
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": repository, "revision": revision}
    if local_files_only:
        kwargs["local_files_only"] = True
    path = Path(snapshot_download(**kwargs))
    resolved = path.resolve()
    if resolved.name != revision:
        raise MacASRConfigurationError(
            f"Resolved MLX snapshot {resolved.name!r} does not match {revision!r}"
        )
    return str(resolved)


def cached_snapshot_path(repository, revision):
    """Return a pinned local snapshot path without downloading anything."""
    return _snapshot_path(repository, revision, local_files_only=True)


def _import_mlx_package(name):
    """Avoid re-importing MLX after a native-extension initialization failure."""
    global _mlx_import_error
    if _mlx_import_error is not None:
        raise RuntimeError(
            f"MLX initialization previously failed: {_mlx_import_error}"
        )
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        _mlx_import_error = f"{type(error).__name__}: {error}"
        raise RuntimeError(
            f"{name} is unavailable; install requirements-macos.txt"
        ) from error


def _mlx_kwargs(params):
    """Translate the stable OpenShorts Whisper settings to mlx-whisper."""
    translated = {
        "verbose": None,
        "word_timestamps": bool(params.get("word_timestamps", True)),
        "condition_on_previous_text": bool(
            params.get("condition_on_previous_text", False)
        ),
    }
    if params.get("language"):
        translated["language"] = params["language"]
    # mlx-whisper 0.4.3 implements greedy decoding only: passing a
    # ``beam_size`` (including 1) raises NotImplementedError. Keep the
    # upstream faster-whisper setting on its own route, but omit it here so
    # the Mac adapter can use mlx-whisper's supported greedy decoder.
    # mlx-whisper has no Silero ``vad_filter`` argument.  Its native
    # no-speech filtering remains enabled for both full media and repair spans.
    return translated


def _normalise_result(result):
    from subtitles import merge_continuation_words

    segments = []
    for source in result.get("segments") or []:
        words = []
        for item in source.get("words") or []:
            if item.get("start") is None or item.get("end") is None:
                continue
            start = float(item["start"])
            end = max(start, float(item["end"]))
            text = str(item.get("word") or "")
            if not words and text and not text.startswith(" "):
                text = " " + text
            words.append({"word": text, "start": start, "end": end})
        words.sort(key=lambda word: (word["start"], word["end"]))
        segments.append({
            "start": float(source.get("start", 0.0)),
            "end": float(source.get("end", 0.0)),
            "text": str(source.get("text") or ""),
            "words": merge_continuation_words(words),
        })
    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    text = str(result.get("text") or "").strip()
    if not text:
        text = " ".join(
            segment["text"].strip()
            for segment in segments
            if segment["text"].strip()
        )
    return {
        "text": text,
        "language": str(result.get("language") or "en"),
        "segments": segments,
    }


def transcribe_whisper(audio, **params):
    """Run pinned mlx-whisper and return the OpenShorts transcript contract."""
    mlx_whisper = _import_mlx_package("mlx_whisper")

    repository, revision = resolve_whisper_model()
    model_path = _snapshot_path(repository, revision)
    with _whisper_lock:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_path,
            **_mlx_kwargs(params),
        )
    if not isinstance(result, dict):
        raise RuntimeError("mlx-whisper returned a non-object result")
    return _normalise_result(result)


def _normalise_parakeet_result(result):
    from subtitles import merge_continuation_words

    segments = []
    for source in getattr(result, "sentences", None) or []:
        words = []
        for item in getattr(source, "tokens", None) or []:
            if getattr(item, "start", None) is None:
                continue
            start = float(item.start)
            end = max(start, float(getattr(item, "end", start)))
            text = str(getattr(item, "text", "") or "")
            if not words and text and not text.startswith(" "):
                text = " " + text
            if text:
                words.append({"word": text, "start": start, "end": end})
        words.sort(key=lambda word: (word["start"], word["end"]))
        words = merge_continuation_words(words)
        if not words:
            continue
        text = str(getattr(source, "text", "") or "").strip()
        segments.append({
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text or "".join(word["word"] for word in words).strip(),
            "words": words,
        })
    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments)
    return {"text": text, "segments": segments}


def transcribe_parakeet(audio, *, chunk_callback=None):
    """Run pinned parakeet-mlx and return normalized text and segments."""
    parakeet_mlx = _import_mlx_package("parakeet_mlx")

    repository, revision = resolve_parakeet_model()
    model_path = _snapshot_path(repository, revision)
    key = (repository, revision)
    global _parakeet_model, _parakeet_key
    with _parakeet_lock:
        if _parakeet_model is None or _parakeet_key != key:
            _parakeet_model = parakeet_mlx.from_pretrained(model_path)
            _parakeet_key = key
        result = _parakeet_model.transcribe(
            audio,
            chunk_duration=120,
            overlap_duration=15,
            chunk_callback=chunk_callback,
        )
    return _normalise_parakeet_result(result)


def audio_duration_seconds(audio):
    """Best-effort duration for existing progress logging."""
    if not isinstance(audio, (str, os.PathLike)):
        try:
            return max(0.0, float(len(audio)) / 16000.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", os.fspath(audio),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return max(0.0, float(result.stdout.strip()))
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def release_whisper_model():
    """Release mlx-whisper state without importing MLX just to clear it."""
    released = False
    with _whisper_lock:
        transcribe_module = sys.modules.get("mlx_whisper.transcribe")
        holder = getattr(transcribe_module, "ModelHolder", None)
        if holder is not None:
            released = getattr(holder, "model", None) is not None
            holder.model = None
            holder.model_path = None
    core = sys.modules.get("mlx.core")
    clear_cache = getattr(core, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    return released


def release_models():
    """Release both MLX ASR engines, then clear the shared MLX cache."""
    global _parakeet_model, _parakeet_key
    released = False
    with _whisper_lock:
        transcribe_module = sys.modules.get("mlx_whisper.transcribe")
        holder = getattr(transcribe_module, "ModelHolder", None)
        if holder is not None:
            released = getattr(holder, "model", None) is not None
            holder.model = None
            holder.model_path = None
    with _parakeet_lock:
        released = released or _parakeet_model is not None
        _parakeet_model = None
        _parakeet_key = None
    core = sys.modules.get("mlx.core")
    clear_cache = getattr(core, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    return released
