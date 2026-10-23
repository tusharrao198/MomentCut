"""Safe, shared metadata helpers for generated short clips.

This module has no network or provider dependency.  It normalises optional
packaging fields returned by Qwen/Gemini, composes a YouTube description with a
local channel profile, and extracts only a selected clip's transcript context.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


DESCRIPTION_LIMIT_BYTES = 5_000
MODEL_DESCRIPTION_LIMIT_BYTES = 1_200
TAG_LIMIT_BYTES = 60
TAG_LIST_LIMIT_CHARS = 500
MAX_MODEL_TAGS = 15
MAX_TOTAL_TAGS = 20
MAX_HASHTAGS = 5
MAX_PROFILE_BYTES = 64 * 1024
MAX_PROFILE_SECTIONS = 12
MAX_PROFILE_SECTION_BYTES = 1_500
MAX_BRIEF_VALUE_BYTES = 240
MAX_BRIEF_ITEM_BYTES = 90
MAX_BRIEF_ITEMS = 6
MAX_THUMBNAIL_TEXT_CHARS = 40

_BRIEF_KEYS = (
    "subject",
    "moment",
    "emotion",
    "visual_style",
    "composition",
    "thumbnail_text",
)
_PROFILE_KEYS = frozenset(("default_tags", "default_hashtags", "description_sections"))
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _utf8_limit(value: str, maximum: int) -> str:
    """Truncate string at a UTF-8 boundary."""
    encoded = value.encode("utf-8", errors="ignore")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore").rstrip()


def _clean_text(value: Any, maximum: int, *, preserve_newlines: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    value = _CONTROL_RE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = value.replace("<", "").replace(">", "")
    if preserve_newlines:
        lines = [" ".join(line.split()) for line in value.split("\n")]
        # Keep paragraphs, but prevent a model/profile from creating a huge
        # vertical gap in a YouTube description.
        output: list[str] = []
        prior_blank = False
        for line in lines:
            blank = not line
            if blank and prior_blank:
                continue
            output.append(line)
            prior_blank = blank
        value = "\n".join(output).strip()
    else:
        value = " ".join(value.split())
    return _utf8_limit(value, maximum)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def normalise_tags(value: Any, *, maximum_items: int = MAX_MODEL_TAGS) -> list[str]:
    """Return specific, non-hashtag tags with a conservative YouTube-safe cap."""
    output: list[str] = []
    seen: set[str] = set()
    for raw in _iter_strings(value):
        tag = _clean_text(raw, TAG_LIMIT_BYTES)
        tag = tag.lstrip("#").strip()
        if not tag:
            continue
        folded = tag.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        output.append(tag)
        if len(output) >= maximum_items:
            break
    return output


def normalise_hashtags(value: Any, *, maximum_items: int = MAX_HASHTAGS) -> list[str]:
    """Return distinct single-token hashtags while preserving Unicode letters."""
    output: list[str] = []
    seen: set[str] = set()
    for raw in _iter_strings(value):
        tag = _clean_text(raw, TAG_LIMIT_BYTES)
        tag = "#" + re.sub(r"[\s#]+", "", tag.lstrip("#"))
        if tag == "#":
            continue
        folded = tag.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        output.append(tag)
        if len(output) >= maximum_items:
            break
    return output


def normalise_thumbnail_brief(value: Any) -> dict:
    """Make optional model packaging output bounded and renderer safe."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    source = value if isinstance(value, dict) else {}
    result = {
        key: _clean_text(source.get(key), MAX_BRIEF_VALUE_BYTES)
        for key in _BRIEF_KEYS
    }
    result["thumbnail_text"] = result["thumbnail_text"][:MAX_THUMBNAIL_TEXT_CHARS]
    for key in ("must_include", "avoid"):
        items: list[str] = []
        for raw in _iter_strings(source.get(key)):
            clean = _clean_text(raw, MAX_BRIEF_ITEM_BYTES)
            if clean and clean.casefold() not in {item.casefold() for item in items}:
                items.append(clean)
            if len(items) >= MAX_BRIEF_ITEMS:
                break
        result[key] = items
    return result


def normalise_clip_packaging_metadata(clip: dict) -> dict:
    """Copy ``clip`` and safely add optional YouTube/thumbnail metadata."""
    result = dict(clip) if isinstance(clip, dict) else {}
    result["youtube_description_intro"] = _clean_text(
        result.get("youtube_description_intro"),
        MODEL_DESCRIPTION_LIMIT_BYTES,
        preserve_newlines=True,
    )
    result["youtube_tags"] = normalise_tags(result.get("youtube_tags"))
    result["source_youtube_tags"] = normalise_tags(
        result.get("source_youtube_tags"), maximum_items=MAX_TOTAL_TAGS
    )
    result["youtube_hashtags"] = normalise_hashtags(result.get("youtube_hashtags"))
    result["thumbnail_brief"] = normalise_thumbnail_brief(result.get("thumbnail_brief"))
    result["why"] = _clean_text(result.get("why"), 320)
    return result


def conservative_thumbnail_brief(clip: dict, transcript_excerpt: str) -> dict:
    """Fallback when an older/smaller model omitted all thumbnail details."""
    brief = normalise_thumbnail_brief((clip or {}).get("thumbnail_brief"))
    if any(brief.get(key) for key in _BRIEF_KEYS) or brief["must_include"] or brief["avoid"]:
        return brief
    title = _clean_text((clip or {}).get("video_title_for_youtube_short"), 160)
    hook = _clean_text((clip or {}).get("viral_hook_text"), 160)
    context = _clean_text(transcript_excerpt, 240)
    return {
        "subject": title or hook or context,
        "moment": hook or title or context,
        "emotion": "",
        "visual_style": "high-contrast cinematic YouTube thumbnail",
        "composition": "clear subject with uncluttered negative space for short text",
        "thumbnail_text": hook[:MAX_THUMBNAIL_TEXT_CHARS],
        "must_include": [],
        "avoid": [],
    }


def _empty_profile() -> dict:
    return {"default_tags": [], "default_hashtags": [], "description_sections": []}


def _profile_path() -> tuple[Path | None, str | None]:
    root = (Path(__file__).resolve().parent / "config").resolve()
    raw = (os.environ.get("YOUTUBE_CHANNEL_PROFILE_PATH") or "").strip()
    candidate = Path(raw) if raw else root / "youtube-channel-profile.json"
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None, "YouTube channel profile path must be inside OpenShorts/config."
    return resolved, None


def load_youtube_channel_profile() -> tuple[dict, str | None]:
    """Load a small, non-secret optional profile without arbitrary file reads."""
    path, problem = _profile_path()
    if problem:
        return _empty_profile(), problem
    if path is None or not path.exists():
        return _empty_profile(), None
    try:
        if not path.is_file() or path.stat().st_size > MAX_PROFILE_BYTES:
            return _empty_profile(), "YouTube channel profile is invalid or larger than 64 KiB."
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return _empty_profile(), "YouTube channel profile could not be read; using clip metadata only."
    if not isinstance(data, dict) or set(data) - _PROFILE_KEYS:
        return _empty_profile(), "YouTube channel profile has unsupported fields; using clip metadata only."
    if not all(isinstance(data.get(key, []), list) for key in _PROFILE_KEYS):
        return _empty_profile(), "YouTube channel profile values must be lists; using clip metadata only."
    return {
        "default_tags": normalise_tags(data.get("default_tags"), maximum_items=MAX_TOTAL_TAGS),
        "default_hashtags": normalise_hashtags(data.get("default_hashtags")),
        "description_sections": [
            section for section in (
                _clean_text(item, MAX_PROFILE_SECTION_BYTES, preserve_newlines=True)
                for item in _iter_strings(data.get("description_sections"))
            )
            if section
        ][:MAX_PROFILE_SECTIONS],
    }, None


def _fit_youtube_tags(tags: Iterable[str]) -> list[str]:
    """Keep the serialized YouTube tag list within its 500-character budget."""
    output: list[str] = []
    used = 0
    for tag in normalise_tags(list(tags), maximum_items=MAX_TOTAL_TAGS):
        # YouTube's documented character budget includes separators and quotes
        # around tags containing spaces.
        encoded = len(tag) + (2 if " " in tag else 0) + (1 if output else 0)
        if used + encoded > TAG_LIST_LIMIT_CHARS:
            break
        output.append(tag)
        used += encoded
    return output


def compose_youtube_metadata(clip: dict) -> tuple[str, list[str], bool, str | None]:
    """Compose the description/tags to hand to Postiz for one clip."""
    normalised = normalise_clip_packaging_metadata(clip)
    profile, warning = load_youtube_channel_profile()
    hashtags = normalise_hashtags(
        normalised["youtube_hashtags"] + profile["default_hashtags"]
    )
    source_tag_line = (
        "Source video tags: " + ", ".join(normalised["source_youtube_tags"])
        if normalised["source_youtube_tags"] else ""
    )
    sections = [
        normalised["youtube_description_intro"],
        source_tag_line,
        " ".join(hashtags),
        *profile["description_sections"],
    ]
    description = "\n\n".join(section for section in sections if section).strip()
    truncated = len(description.encode("utf-8")) > DESCRIPTION_LIMIT_BYTES
    description = _utf8_limit(description, DESCRIPTION_LIMIT_BYTES)
    tags = _fit_youtube_tags(normalised["youtube_tags"] + profile["default_tags"])
    return description, tags, truncated, warning


def clip_transcript_excerpt(transcript: Any, start: Any, end: Any) -> str:
    """Return only transcript words overlapping one current clip."""
    try:
        start_seconds, end_seconds = float(start), float(end)
    except (TypeError, ValueError):
        return ""
    if end_seconds <= start_seconds or not isinstance(transcript, dict):
        return ""
    parts: list[str] = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                try:
                    overlaps = float(word.get("end", 0)) > start_seconds and float(word.get("start", 0)) < end_seconds
                except (TypeError, ValueError):
                    overlaps = False
                if overlaps and isinstance(word.get("word"), str):
                    parts.append(word["word"].strip())
            continue
        try:
            overlaps = float(segment.get("end", 0)) > start_seconds and float(segment.get("start", 0)) < end_seconds
        except (TypeError, ValueError):
            overlaps = False
        if overlaps and isinstance(segment.get("text"), str):
            parts.append(segment["text"].strip())
    return _clean_text(" ".join(part for part in parts if part), MODEL_DESCRIPTION_LIMIT_BYTES)
