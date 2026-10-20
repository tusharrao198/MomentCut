"""Decode with NVDEC or VideoToolbox, then return CPU-readable frames.

A job decodes each clip several times: scene detection, the face classifier,
the split detector, the camera trajectory, every reframe segment and the
caption/hook burn. On the CPU those decodes were the largest share of a job's
CPU time while the card's decoder sat idle.

H.264/HEVC decoding is exact, so NVDEC returns the same frames as libavcodec.
What changes is the pixel format they arrive in (nv12 instead of yuv420p),
and swscale converts nv12 by another path: an nv12 frame turned straight
into BGR differs from the CPU decode. ``DOWNLOAD`` therefore brings each frame
back as nv12 and repacks it to yuv420p (a lossless copy of the same samples)
before any other filter, and from there the graph is the one the CPU path
runs: frames, samples and encodes come out byte-identical.

Only plain 8-bit 4:2:0 H.264/HEVC without a rotation goes to hardware. NVIDIA
keeps its existing free-VRAM gate and CUDA/NVDEC command. Apple Silicon probes
the actual source with a forced VideoToolbox hardware-frame download, so an
ffmpeg that silently falls back to software is rejected. Any real hardware
operation that fails is replayed on the CPU by its owning consumer.

``VIDEO_DECODE_BACKEND=auto`` selects CUDA off Apple when the existing NVIDIA
gate passes, VideoToolbox on Apple Silicon when its source probe passes, and
CPU otherwise. Explicit ``cuda``, ``videotoolbox`` and ``cpu`` values are
diagnostic overrides with the same safe fallback. ``GPU_DECODE=0`` turns
hardware decoding off; ``GPU_DECODE_RENDER=0`` keeps hardware analysis but
decodes render operations on the CPU.
"""
from dataclasses import dataclass
import json
import os
import subprocess
import threading
import time

from accelerator_runtime import is_apple_silicon

ENABLED = os.environ.get("GPU_DECODE", "1").strip() != "0"
RENDER = ENABLED and os.environ.get("GPU_DECODE_RENDER", "1").strip() != "0"
MIN_FREE_MB = int(os.environ.get("GPU_DECODE_MIN_FREE_MB", "2500") or 0)

HW_ARGS = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
DOWNLOAD = "hwdownload,format=nv12,format=yuv420p"
VIDEOTOOLBOX_HW_ARGS = [
    "-hwaccel", "videotoolbox",
    "-hwaccel_output_format", "videotoolbox_vld",
]
_CODECS = {"h264", "hevc"}
_BACKENDS = {"auto", "cuda", "videotoolbox", "cpu"}

_lock = threading.Lock()
_probe_cache = {}
_videotoolbox_probe_cache = {}
_free = {"t": 0.0, "mb": None}
STATS = {"gpu": 0, "fallback": 0, "skipped": 0}


class DecodeConfigurationError(ValueError):
    """The requested decode backend is not a supported configuration."""


@dataclass(frozen=True)
class DecodeAttempt:
    backend: str
    input_args: tuple
    download_filter: str
    is_hardware: bool
    render: bool


def _attempt(backend, render=False):
    if backend == "cuda":
        return DecodeAttempt(
            backend, tuple(HW_ARGS), DOWNLOAD, True, bool(render))
    if backend == "videotoolbox":
        return DecodeAttempt(
            backend, tuple(VIDEOTOOLBOX_HW_ARGS), DOWNLOAD, True, bool(render))
    return DecodeAttempt("cpu", (), "", False, bool(render))


def _as_attempt(value, render=False):
    """Accept the N4 attempt object and the old CUDA boolean API."""
    if isinstance(value, DecodeAttempt):
        return value
    return _attempt("cuda" if value else "cpu", render=render)


def _count(key):
    with _lock:
        STATS[key] += 1


def _probe(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    with _lock:
        if key in _probe_cache:
            return _probe_cache[key]
    info = None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=index,codec_name,profile,pix_fmt,width,height:"
             "stream_side_data=rotation",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        streams = json.loads(out.stdout or "{}").get("streams") or []
        if streams:
            info = streams[0]
    except Exception:
        info = None
    with _lock:
        _probe_cache[key] = info
    return info


def _supported(info):
    if not info:
        return False
    if info.get("codec_name") not in _CODECS or info.get("pix_fmt") != "yuv420p":
        return False
    for side in info.get("side_data_list") or []:
        if int(float(side.get("rotation") or 0)) % 360:
            return False
    return True


def _free_mb():
    """Free card memory in MB, refreshed at most every 2 s (None: unknown)."""
    now = time.time()
    with _lock:
        if now - _free["t"] < 2:
            return _free["mb"]
    mb = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            mb = int(out.stdout.split()[0])
    except Exception:
        mb = None
    with _lock:
        _free.update(t=now, mb=mb)
    return mb


def usable(path, render=False):
    """True when ``path`` should be decoded on the card right now."""
    if not (RENDER if render else ENABLED):
        return False
    if not _supported(_probe(path)):
        return False
    free = _free_mb()
    if free is None or free < MIN_FREE_MB:
        _count("skipped")
        return False
    return True


def _videotoolbox_probe_key(path, info):
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


def _probe_videotoolbox(path, info):
    """Prove that the actual source produced a VideoToolbox hardware frame.

    FFmpeg may report a VideoToolbox initialization error, decode in software
    and still exit zero. The forced ``hwdownload`` makes that fallback fail,
    while the verbose filter input supplies positive hardware-surface evidence.
    """
    key = _videotoolbox_probe_key(path, info)
    if not key:
        return False
    with _lock:
        if key in _videotoolbox_probe_cache:
            return _videotoolbox_probe_cache[key]

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "verbose",
        *VIDEOTOOLBOX_HW_ARGS, "-i", path,
        "-map", "0:v:0", "-frames:v", "1",
        "-vf", DOWNLOAD, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
    ]
    ok = False
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
        _videotoolbox_probe_cache[key] = ok
    return ok


def videotoolbox_usable(path, render=False):
    if not (RENDER if render else ENABLED):
        return False
    if not is_apple_silicon():
        return False
    info = _probe(path)
    if not _supported(info):
        return False
    if not _probe_videotoolbox(path, info):
        _count("skipped")
        return False
    return True


def select_attempt(path=None, render=False, backend=None):
    """Choose a source-qualified decode attempt without changing CUDA policy."""
    requested = (
        os.environ.get("VIDEO_DECODE_BACKEND", "auto")
        if backend is None else backend
    )
    requested = str(requested or "auto").strip().lower()
    if requested not in _BACKENDS:
        raise DecodeConfigurationError(
            "VIDEO_DECODE_BACKEND must be auto, cuda, videotoolbox, or cpu")
    if not (RENDER if render else ENABLED) or requested == "cpu" or not path:
        return _attempt("cpu", render)
    if requested == "auto":
        requested = "videotoolbox" if is_apple_silicon() else "cuda"
    if requested == "videotoolbox":
        return (
            _attempt("videotoolbox", render)
            if videotoolbox_usable(path, render=render)
            else _attempt("cpu", render)
        )
    return (
        _attempt("cuda", render)
        if usable(path, render=render)
        else _attempt("cpu", render)
    )


def input_args(attempt):
    """Options to put right before the video's ``-i``."""
    return list(_as_attempt(attempt).input_args)


def download_filter(attempt, target_format="yuv420p"):
    selected = _as_attempt(attempt)
    if not selected.is_hardware:
        return ""
    if not target_format or target_format == "yuv420p":
        return selected.download_filter
    return f"{selected.download_filter},format={target_format}"


def vf(chain, attempt):
    """A -vf chain preceded by hardware download when needed."""
    selected = _as_attempt(attempt)
    if not selected.is_hardware:
        return chain
    bridge = download_filter(selected)
    return f"{bridge},{chain}" if chain else bridge


def graph(filter_complex, attempt, label="[0:v]"):
    """A graph whose video input ``label`` is downloaded from hardware.

    The first use of ``label`` is fed by the download instead; a graph that
    names it more than once (which ffmpeg would reject anyway) or not at all is
    returned untouched, and so is the decode for it (see ``graph_ok``)."""
    selected = _as_attempt(attempt)
    if not selected.is_hardware:
        return filter_complex
    return (
        f"{label}{download_filter(selected)}[gpudec];"
        + filter_complex.replace(label, "[gpudec]", 1)
    )


def graph_ok(filter_complex, label="[0:v]"):
    return filter_complex.count(label) == 1


def note(attempt, failed=False):
    selected = _as_attempt(attempt)
    if failed:
        _count("fallback")
    elif selected.is_hardware:
        _count("gpu")


def run_with_backend(build, runner, ok=lambda result: result.returncode == 0,
                     path=None, render=True, backend=None,
                     before_cpu_retry=None, retry_videotoolbox=True):
    """Run one backend-aware operation and replay a failed hardware attempt.

    ``before_cpu_retry`` is used by file-producing VideoToolbox consumers to
    remove partial output. CUDA keeps its established retry behavior unchanged.
    A combined VideoToolbox decode/encode command can defer replay to
    ``ffmpeg_utils.run_encode``, which owns the required CPU+x264 transaction.
    """
    attempt = select_attempt(path, render=render, backend=backend)
    cpu = _attempt("cpu", render)
    command = build(attempt)
    if attempt.is_hardware and command == build(cpu):
        attempt = cpu

    try:
        result = runner(command)
    except Exception:
        if not attempt.is_hardware:
            raise
        if attempt.backend == "cuda":
            raise
        note(attempt, failed=True)
        if attempt.backend == "videotoolbox" and not retry_videotoolbox:
            raise
        if attempt.backend == "videotoolbox" and before_cpu_retry:
            before_cpu_retry()
        print(f"   ⚠️ {attempt.backend} decode failed, decoding on the CPU")
        return runner(build(cpu))

    if attempt.is_hardware and not ok(result):
        note(attempt, failed=True)
        if attempt.backend == "videotoolbox" and not retry_videotoolbox:
            return result
        if attempt.backend == "videotoolbox" and before_cpu_retry:
            before_cpu_retry()
        print(f"   ⚠️ {attempt.backend} decode failed, decoding on the CPU")
        return runner(build(cpu))
    note(attempt)
    return result


def run_command(command, runner, ok=lambda result: result.returncode == 0,
                path=None, render=True, backend=None,
                before_cpu_retry=None, retry_videotoolbox=True):
    """Backend-aware execution for an already-built filtered FFmpeg command."""
    return run_with_backend(
        lambda attempt: on_card(command, path, attempt),
        runner,
        ok=ok,
        path=path,
        render=render,
        backend=backend,
        before_cpu_retry=before_cpu_retry,
        retry_videotoolbox=retry_videotoolbox,
    )


def run(build, runner, ok=lambda result: result.returncode == 0, path=None, render=True):
    """Run ``runner(build(on_gpu))``; if the card was used and the result is not
    ``ok``, run it again on the CPU. Returns the last result."""
    on_gpu = usable(path, render=render) if path else False
    cmd = build(on_gpu)
    if on_gpu and cmd == build(False):
        on_gpu = False  # nothing to put on the card
    result = runner(cmd)
    if on_gpu and not ok(result):
        note(True, failed=True)
        print("   ⚠️ GPU decode failed, decoding on the CPU")
        return runner(build(False))
    note(on_gpu)
    return result


def on_card(cmd, video_path, attempt=None):
    """``cmd`` with ``video_path`` decoded by the selected hardware backend.

    With no explicit attempt this remains the original CUDA compatibility
    wrapper. The hwaccel options go
    before its ``-i`` (which must be the first input) and the download heads
    its -vf / -filter_complex. ``cmd`` itself when that cannot be done safely
    (another first input, no filter, a graph naming [0:v] more than once)."""
    selected = _attempt("cuda") if attempt is None else _as_attempt(attempt)
    if not selected.is_hardware:
        return cmd
    try:
        first = cmd.index("-i")
    except ValueError:
        return cmd
    if cmd[first + 1] != video_path:
        return cmd
    out = list(cmd)
    if "-filter_complex" in out:
        k = out.index("-filter_complex") + 1
        if not graph_ok(out[k]):
            return cmd
        out[k] = graph(out[k], selected)
    elif "-vf" in out and out.count("-vf") == 1:
        k = out.index("-vf") + 1
        out[k] = vf(out[k], selected)
    else:
        return cmd
    return out[:first] + input_args(selected) + out[first:]


# How many frames ffmpeg decoded from a clip in the scene pass, keyed by the
# file's identity. The camera trajectory needs that count to decode only the
# frames it looks at (reframe_v2._sparse_trajectory).
_decoded = {}


def _file_key(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)


def remember_count(path, n):
    key = _file_key(path)
    if key:
        with _lock:
            _decoded[key] = n


def any_of(terms):
    """ffmpeg expression true when any of ``terms`` is, as a balanced tree of
    sums: ffmpeg refuses a flat a+b+c+... of more than about a hundred terms
    ("Cannot allocate memory" from its expression parser's depth limit), and a
    clip easily asks for more samples than that."""
    terms = list(terms)
    if not terms:
        return "0"
    while len(terms) > 1:
        terms = [f"({terms[k]}+{terms[k + 1]})" if k + 1 < len(terms) else terms[k]
                 for k in range(0, len(terms), 2)]
    return terms[0]


def one_frame_per_decoded_frame(stderr):
    """True when an ffmpeg rawvideo pipe carried the decoded frames one for one.

    A rawvideo output runs ffmpeg's constant-frame-rate sync, which repeats or
    drops frames when the timestamps are irregular; its final report then
    shows ``dup=``/``drop=``. Only without those is frame k of the pipe the
    k-th decoded frame, which is what the sparse trajectory pass relies on."""
    text = (stderr or b"").decode(errors="replace")
    return "frame=" in text and "dup=" not in text and "drop=" not in text


def decoded_frames(path):
    """Frames ffmpeg decoded from this exact file in the scene pass, or None."""
    key = _file_key(path)
    if not key:
        return None
    with _lock:
        return _decoded.get(key)
