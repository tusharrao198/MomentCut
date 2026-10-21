"""Apple VideoToolbox capability checks and command fragments.

The shared encode/decode modules ask this adapter only after
``APPLE_SILICON_ACCELERATION`` selected an active Apple route.  It contains no
OpenShorts queueing, validation, or retry policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import threading

VIDEOTOOLBOX = "videotoolbox"
DOWNLOAD = "hwdownload,format=nv12,format=yuv420p"
HW_ARGS = (
    "-hwaccel", "videotoolbox",
    "-hwaccel_output_format", "videotoolbox_vld",
)

_ENCODE_ARGS = {
    "quality": ["-c:v", "h264_videotoolbox", "-q:v", "75",
                "-allow_sw", "0", "-pix_fmt", "yuv420p"],
    "quality_fast": ["-c:v", "h264_videotoolbox", "-q:v", "75",
                     "-allow_sw", "0", "-pix_fmt", "yuv420p"],
    "delivery": ["-c:v", "h264_videotoolbox", "-q:v", "65",
                 "-allow_sw", "0", "-pix_fmt", "yuv420p"],
}
_CODECS = {"h264", "hevc"}
_lock = threading.Lock()
_encoder_available = None
_decode_probe_cache = {}


@dataclass(frozen=True)
class EncoderDecision:
    """A platform answer consumed by shared encoder policy."""

    encoder: str
    warning: str = ""


@dataclass(frozen=True)
class DecodePlan:
    """A platform decode choice consumed as a shared DecodeAttempt."""

    backend: str
    input_args: tuple
    download_filter: str
    is_hardware: bool


def _probe_encoder() -> bool:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-frames:v", "1", "-c:v", "h264_videotoolbox",
        "-allow_sw", "0", "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    try:
        return subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).returncode == 0
    except Exception:
        return False


def encoder_available() -> bool:
    """Probe VideoToolbox encode once for this process."""
    global _encoder_available
    if _encoder_available is None:
        with _lock:
            if _encoder_available is None:
                _encoder_available = _probe_encoder()
    return _encoder_available


def encoder_decision(tier: str, mode: str):
    """Return a decision for Mac ``auto``/``videotoolbox``, else ``None``."""
    if mode not in {"auto", VIDEOTOOLBOX}:
        return None
    # Intermediate cuts are decoded again immediately.  The chosen Mac policy
    # keeps them on the shared CPU/x264 path and avoids an unnecessary probe.
    if tier == "intermediate":
        return EncoderDecision("x264")
    if encoder_available():
        return EncoderDecision(VIDEOTOOLBOX)
    if mode == VIDEOTOOLBOX:
        return EncoderDecision(
            "x264",
            "⚠️ [Encoder] FFMPEG_ENCODER=videotoolbox but "
            "h264_videotoolbox is not usable here — falling back to libx264",
        )
    return EncoderDecision("x264")


def encoder_args(tier: str) -> list[str]:
    """Return a fresh VideoToolbox codec/quality argv fragment."""
    try:
        return list(_ENCODE_ARGS[tier])
    except KeyError as error:
        raise ValueError(f"VideoToolbox does not support encode tier: {tier!r}") from error


def disable_encoder() -> None:
    """Disable VideoToolbox after its shared operation replay succeeds."""
    global _encoder_available
    with _lock:
        _encoder_available = False


def reset_encoder_cache() -> None:
    """Test hook: forget VideoToolbox capability state."""
    global _encoder_available
    with _lock:
        _encoder_available = None
        _decode_probe_cache.clear()


def _file_key(path: str):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return os.path.realpath(path), stat.st_size, stat.st_mtime_ns


def _source_info(path: str):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=index,codec_name,profile,pix_fmt,width,height:"
                "stream_side_data=rotation",
                "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(result.stdout or "{}").get("streams") or []
        return streams[0] if streams else None
    except Exception:
        return None


def _supported_source(info) -> bool:
    if not info:
        return False
    if info.get("codec_name") not in _CODECS or info.get("pix_fmt") != "yuv420p":
        return False
    return not any(
        int(float(side.get("rotation") or 0)) % 360
        for side in info.get("side_data_list") or []
    )


def _decode_probe_key(path: str, info):
    key = _file_key(path)
    if not key or not info:
        return None
    rotation = tuple(
        int(float(side.get("rotation") or 0))
        for side in info.get("side_data_list") or []
    )
    return key + (
        info.get("index"),
        info.get("codec_name"),
        info.get("profile"),
        info.get("pix_fmt"),
        info.get("width"),
        info.get("height"),
        rotation,
    )


def _probe_decode(path: str, info) -> bool:
    """Prove the actual source produces VideoToolbox hardware frames."""
    key = _decode_probe_key(path, info)
    if not key:
        return False
    with _lock:
        if key in _decode_probe_cache:
            return _decode_probe_cache[key]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "verbose",
        *HW_ARGS, "-i", path,
        "-map", "0:v:0", "-frames:v", "1",
        "-vf", DOWNLOAD, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        stderr = (result.stderr or b"").decode(errors="replace").lower()
        expected = int(info.get("width") or 0) * int(info.get("height") or 0) * 3 // 2
        rejected = (
            "videotoolbox malfunction",
            "failed setup for format videotoolbox_vld",
            "hwaccel initialisation returned error",
            "videotoolbox session not available",
            "videotoolbox does not support this format",
            "videotoolbox decoder for this format not found",
        )
        ok = (
            result.returncode == 0
            and expected > 0
            and len(result.stdout or b"") >= expected
            and "pixfmt:videotoolbox_vld" in stderr
            and not any(term in stderr for term in rejected)
        )
    except Exception:
        ok = False
    with _lock:
        _decode_probe_cache[key] = ok
    return ok


def decode_plan(path: str, *, enabled: bool):
    """Return a hardware plan for explicit VideoToolbox, or ``None``."""
    if not enabled:
        return None
    info = _source_info(path)
    if not _supported_source(info) or not _probe_decode(path, info):
        return None
    return DecodePlan(VIDEOTOOLBOX, HW_ARGS, DOWNLOAD, True)


def precise_cut_command(input_video, start, end, encode_args, audio_args, output,
                        decode_attempt):
    """Build the only VideoToolbox-specific clean-cut command shape."""
    return [
        "ffmpeg", "-y", *decode_attempt.input_args,
        "-ss", str(start), "-to", str(end), "-i", input_video,
        "-vf", decode_attempt.download_filter,
        *encode_args, *audio_args, output,
    ]
