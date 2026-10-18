"""Explicit opt-ins for Gemini features that are separate from clip selection.

Having ``GEMINI_API_KEY`` configured must not by itself make a local-first
pipeline send video frames to Gemini.  These flags keep local Qwen/Qwen-VL
selection independent from optional Gemini helpers and the user-triggered
Nano Banana thumbnail action.
"""
from __future__ import annotations

import os


_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def _enabled(name: str, default: bool) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def clip_analysis_enabled() -> bool:
    """Whether automatic Gemini calls are allowed during clip processing.

    ``True`` preserves upstream behaviour for installations which use Gemini
    for transcript selection, silent-video fallback, layout selection, or hook
    grounding. Local-first installations set this to ``0``.
    """
    return _enabled("GEMINI_CLIP_ANALYSIS_ENABLED", True)


def clip_ai_thumbnails_enabled() -> bool:
    """Whether the explicit per-clip Nano Banana action is available."""
    return _enabled("CLIP_AI_THUMBNAILS_ENABLED", True)
