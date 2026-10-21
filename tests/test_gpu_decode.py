"""NVDEC decoding (gpu_decode) and the sparse trajectory pass (reframe_v2).

The card itself is not available in CI: these tests pin the command rewrites,
the CPU fallback, and that the sparse trajectory hands the detectors exactly
the frames the full pass did and ends with the same crop trajectory.
"""
import re
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import gpu_decode
import reframe_v2
from platforms.macos import ffmpeg as macos_ffmpeg

CPU = gpu_decode._attempt("cpu")
CUDA = gpu_decode._attempt("cuda")
VIDEOTOOLBOX = gpu_decode.DecodeAttempt(
    "videotoolbox", macos_ffmpeg.HW_ARGS, macos_ffmpeg.DOWNLOAD, True, False)


@pytest.fixture(autouse=True)
def _default_apple_gate(monkeypatch):
    monkeypatch.delenv("APPLE_SILICON_ACCELERATION", raising=False)


# --- command rewrites -------------------------------------------------------

def test_graph_feeds_the_first_input_from_the_card():
    g = gpu_decode.graph("[0:v]split=2[a][b];[a][b]vstack[v]", True)
    assert g == ("[0:v]hwdownload,format=nv12,format=yuv420p[gpudec];"
                 "[gpudec]split=2[a][b];[a][b]vstack[v]")
    assert gpu_decode.graph("[0:v]scale=2:2[v]", False) == "[0:v]scale=2:2[v]"


def test_vf_chain_starts_with_the_download():
    assert gpu_decode.vf("scale=48:27", True) == f"{gpu_decode.DOWNLOAD},scale=48:27"
    assert gpu_decode.vf("scale=48:27", False) == "scale=48:27"


def test_videotoolbox_attempt_uses_the_same_cpu_readable_bridge():
    assert gpu_decode.input_args(VIDEOTOOLBOX) == list(macos_ffmpeg.HW_ARGS)
    assert gpu_decode.vf("scale=48:27", VIDEOTOOLBOX) == (
        f"{gpu_decode.DOWNLOAD},scale=48:27")
    with pytest.raises(FrozenInstanceError):
        VIDEOTOOLBOX.backend = "cpu"


def test_on_card_rewrites_filter_complex_and_input():
    cmd = ["ffmpeg", "-y", "-i", "clip.mp4", "-i", "logo.png",
           "-filter_complex", "[0:v][1:v]overlay=1:2[v]", "-map", "[v]", "out.mp4"]
    out = gpu_decode.on_card(cmd, "clip.mp4")
    assert out[:6] == ["ffmpeg", "-y", *gpu_decode.HW_ARGS]
    assert out[6:8] == ["-i", "clip.mp4"]
    k = out.index("-filter_complex") + 1
    assert out[k].startswith("[0:v]" + gpu_decode.DOWNLOAD + "[gpudec];[gpudec][1:v]overlay")
    # The logo input is left alone.
    assert out[out.index("logo.png") - 1] == "-i"
    assert cmd[2] == "-i"  # the original list is not modified


def test_on_card_rewrites_a_vf_chain():
    cmd = ["ffmpeg", "-y", "-i", "clip.mp4", "-vf", "subtitles=a.ass", "out.mp4"]
    out = gpu_decode.on_card(cmd, "clip.mp4")
    assert out[out.index("-vf") + 1] == f"{gpu_decode.DOWNLOAD},subtitles=a.ass"
    assert out[2:6] == gpu_decode.HW_ARGS


@pytest.mark.parametrize("cmd", [
    ["ffmpeg", "-i", "clip.mp4", "-c", "copy", "out.mp4"],                   # no filter
    ["ffmpeg", "-i", "logo.png", "-i", "clip.mp4", "-vf", "x", "out.mp4"],   # not first
    ["ffmpeg", "-i", "clip.mp4", "-filter_complex",
     "[0:v]split[a];[0:v]null[b]", "out.mp4"],                               # [0:v] twice
])
def test_on_card_leaves_unsafe_commands_alone(cmd):
    assert gpu_decode.on_card(cmd, "clip.mp4") is cmd


def test_run_falls_back_to_the_cpu_when_the_card_fails(monkeypatch):
    monkeypatch.setattr(gpu_decode, "usable", lambda path, render=True: True)
    calls = []

    def runner(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if cmd == "gpu" else 0)

    before = dict(gpu_decode.STATS)
    result = gpu_decode.run(lambda g: "gpu" if g else "cpu", runner, path="x.mp4")
    assert calls == ["gpu", "cpu"]
    assert result.returncode == 0
    assert gpu_decode.STATS["fallback"] == before["fallback"] + 1


def test_run_stays_on_the_cpu_when_the_card_is_not_usable(monkeypatch):
    monkeypatch.setattr(gpu_decode, "usable", lambda path, render=True: False)
    calls = []
    gpu_decode.run(lambda g: "gpu" if g else "cpu",
                   lambda c: calls.append(c) or subprocess.CompletedProcess(c, 0),
                   path="x.mp4")
    assert calls == ["cpu"]


def test_auto_selects_cpu_on_apple_without_probing_hardware(monkeypatch):
    monkeypatch.setattr(gpu_decode.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(
        gpu_decode, "usable",
        lambda path, render=False: pytest.fail("Apple auto must not probe CUDA"))
    monkeypatch.setattr(
        macos_ffmpeg, "decode_plan",
        lambda path, render=False: pytest.fail(
            "Apple auto must not probe VideoToolbox"))
    assert gpu_decode.select_attempt("clip.mp4").backend == "cpu"


def test_explicit_videotoolbox_remains_available_on_apple(monkeypatch):
    monkeypatch.setattr(gpu_decode.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(
        macos_ffmpeg, "decode_plan",
        lambda path, enabled=True: macos_ffmpeg.DecodePlan(
            "videotoolbox", macos_ffmpeg.HW_ARGS, macos_ffmpeg.DOWNLOAD, True))
    monkeypatch.setattr(
        gpu_decode, "usable",
        lambda path, render=False: pytest.fail(
            "Explicit VideoToolbox must not probe CUDA"))
    assert gpu_decode.select_attempt(
        "clip.mp4", backend="videotoolbox").backend == "videotoolbox"


def test_auto_preserves_cuda_selection_off_apple(monkeypatch):
    monkeypatch.setattr(gpu_decode.accelerator_runtime, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(gpu_decode, "usable", lambda path, render=False: True)
    monkeypatch.setattr(
        macos_ffmpeg, "decode_plan",
        lambda path, render=False: pytest.fail("NVIDIA auto must not probe VideoToolbox"))
    assert gpu_decode.select_attempt("clip.mp4").backend == "cuda"


def test_disabled_apple_auto_keeps_cpu_without_hardware_probe(monkeypatch):
    monkeypatch.setattr(gpu_decode.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setenv("APPLE_SILICON_ACCELERATION", "disabled")
    monkeypatch.setattr(
        gpu_decode, "usable",
        lambda *_args, **_kwargs: pytest.fail("disabled Apple must not probe CUDA"))
    monkeypatch.setattr(
        macos_ffmpeg, "decode_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled Apple must not probe VideoToolbox"))
    assert gpu_decode.select_attempt("clip.mp4", backend="auto").backend == "cpu"


def test_explicit_videotoolbox_requires_active_apple_gate(monkeypatch):
    monkeypatch.setattr(gpu_decode.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setenv("APPLE_SILICON_ACCELERATION", "disabled")
    with pytest.raises(gpu_decode.accelerator_runtime.AcceleratorConfigurationError):
        gpu_decode.select_attempt("clip.mp4", backend="videotoolbox")


def test_explicit_cpu_never_probes_hardware(monkeypatch):
    monkeypatch.setattr(
        gpu_decode, "usable",
        lambda path, render=False: pytest.fail("CPU route must not probe CUDA"))
    monkeypatch.setattr(
        macos_ffmpeg, "decode_plan",
        lambda path, render=False: pytest.fail("CPU route must not probe VideoToolbox"))
    assert gpu_decode.select_attempt("clip.mp4", backend="cpu").backend == "cpu"


def test_videotoolbox_probe_rejects_software_fallback_with_exit_zero(
        monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    info = {"index": 0, "codec_name": "h264", "profile": "High",
            "pix_fmt": "yuv420p", "width": 16, "height": 16}
    macos_ffmpeg._decode_probe_cache.clear()
    expected = b"\0" * (16 * 16 * 3 // 2)
    stderr = (
        b"graph input pixfmt:videotoolbox_vld\n"
        b"VideoToolbox malfunction.\n"
        b"Failed setup for format videotoolbox_vld: "
        b"hwaccel initialisation returned error.\n"
    )
    monkeypatch.setattr(
        macos_ffmpeg.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, expected, stderr))
    assert not macos_ffmpeg._probe_decode(str(source), info)


def test_videotoolbox_probe_requires_positive_hardware_surface_and_caches(
        monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    info = {"index": 0, "codec_name": "h264", "profile": "High",
            "pix_fmt": "yuv420p", "width": 16, "height": 16}
    macos_ffmpeg._decode_probe_cache.clear()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, b"\0" * (16 * 16 * 3 // 2),
            b"graph input pixfmt:videotoolbox_vld")

    monkeypatch.setattr(macos_ffmpeg.subprocess, "run", run)
    assert macos_ffmpeg._probe_decode(str(source), info)
    assert macos_ffmpeg._probe_decode(str(source), info)
    assert len(calls) == 1


def test_videotoolbox_probe_cache_includes_stream_properties(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    base = {"index": 0, "codec_name": "h264", "pix_fmt": "yuv420p",
            "width": 16, "height": 16}
    macos_ffmpeg._decode_probe_cache.clear()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, b"\0" * (16 * 16 * 3 // 2),
            b"graph input pixfmt:videotoolbox_vld")

    monkeypatch.setattr(macos_ffmpeg.subprocess, "run", run)
    assert macos_ffmpeg._probe_decode(
        str(source), {**base, "profile": "High"})
    assert macos_ffmpeg._probe_decode(
        str(source), {**base, "profile": "Main"})
    assert len(calls) == 2


def test_videotoolbox_failed_operation_removes_partial_and_replays_cpu(monkeypatch):
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=True, backend=None: (
            CPU if backend == "cpu" else VIDEOTOOLBOX))
    calls = []
    cleanup = []

    result = gpu_decode.run_with_backend(
        lambda attempt: attempt.backend,
        lambda command: (
            calls.append(command),
            subprocess.CompletedProcess(
                command, 1 if command == "videotoolbox" else 0),
        )[1],
        path="clip.mp4",
        before_cpu_retry=lambda: cleanup.append(True),
    )

    assert result.returncode == 0
    assert calls == ["videotoolbox", "cpu"]
    assert cleanup == [True]


def test_combined_videotoolbox_operation_can_defer_complete_replay(monkeypatch):
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=True, backend=None: VIDEOTOOLBOX)
    calls = []
    result = gpu_decode.run_with_backend(
        lambda attempt: attempt.backend,
        lambda command: (
            calls.append(command),
            subprocess.CompletedProcess(command, 1),
        )[1],
        path="clip.mp4",
        retry_videotoolbox=False,
    )
    assert result.returncode == 1
    assert calls == ["videotoolbox"]


def test_new_runner_preserves_cuda_exception_behavior(monkeypatch):
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=True, backend=None: CUDA)
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_decode.run_with_backend(
            lambda attempt: attempt.backend,
            lambda command: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(command, 1)),
            path="clip.mp4",
        )


def test_only_plain_8bit_420_h264_hevc_goes_to_the_card():
    ok = {"codec_name": "h264", "pix_fmt": "yuv420p"}
    assert gpu_decode._supported(ok)
    assert gpu_decode._supported({"codec_name": "hevc", "pix_fmt": "yuv420p"})
    assert not gpu_decode._supported({"codec_name": "h264", "pix_fmt": "yuv420p10le"})
    assert not gpu_decode._supported({"codec_name": "h264", "pix_fmt": "yuvj420p"})
    assert not gpu_decode._supported({"codec_name": "vp9", "pix_fmt": "yuv420p"})
    assert not gpu_decode._supported(
        {**ok, "side_data_list": [{"rotation": -90}]})
    assert gpu_decode._supported({**ok, "side_data_list": [{"rotation": 0}]})
    assert not gpu_decode._supported(None)


# --- select expression -------------------------------------------------------

def _selected(expr, n_max):
    """Evaluate the select= expression the way ffmpeg does, for n < n_max."""
    py = re.sub(r"between\(n,(\d+),(\d+)\)", r"(\1<=n<=\2)", expr)
    py = re.sub(r"not\(mod\(n,(\d+)\)\)", r"(n%\1==0)", py)
    py = re.sub(r"eq\(n,(\d+)\)", r"(n==\1)", py)
    py = re.sub(r"gte\(n,(\d+)\)", r"(n>=\1)", py)
    return [n for n in range(n_max) if eval(py, {"n": n})]


@pytest.mark.parametrize("seed", range(6))
def test_select_expr_is_true_exactly_on_the_frames(seed):
    rng = np.random.default_rng(seed)
    stride_run = list(range(0, 400, 4)) if seed % 2 else []
    frames = sorted(set(stride_run + rng.integers(0, 600, 40).tolist()))
    assert _selected(reframe_v2.select_expr(frames, 4), 700) == frames


def test_select_expr_collapses_stride_runs():
    expr = reframe_v2.select_expr(list(range(0, 4000, 4)), 4)
    assert expr == "between(n,0,3996)*not(mod(n,4))"
    assert reframe_v2.select_expr([1, 5, 6], 4) == "((eq(n,1)+eq(n,5))+eq(n,6))"
    assert reframe_v2.select_expr([], 4) == "0"


# --- sparse trajectory == full trajectory ------------------------------------

class _Cameraman:
    def __init__(self):
        self.current_center_x = 50.0
        self.target_center_x = 50.0
        self.scene_starts = 0

    def begin_scene(self):
        self.scene_starts += 1

    def update_target(self, box):
        self.target_center_x = box[0] + box[2] / 2

    def get_crop_box(self, force_snap=False):
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            self.current_center_x += (self.target_center_x - self.current_center_x) * 0.25
        x = int(self.current_center_x)
        return x, 0, x + 10, 10


class _Tracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_target(self, candidates, frame_number, width):
        return candidates[0]["box"] if candidates else None


@pytest.fixture
def fake_main(monkeypatch):
    seen = []

    def detect_face_candidates(frame):
        n = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        seen.append(("face", n))
        if n % 3 == 0:  # no face on some frames: exercises the YOLO fallback
            return []
        return [{"box": [n % 90, 0, 8, 8]}]

    def detect_person_yolo(frame):
        n = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        seen.append(("yolo", n))
        return [n % 70, 0, 6, 6]

    m = types.SimpleNamespace(
        DETECT_STRIDE=4, YOLO_FALLBACK_STRIDE=8, SCENE_CUT_RESET=True,
        detect_face_candidates=detect_face_candidates,
        detect_person_yolo=detect_person_yolo)
    monkeypatch.setitem(sys.modules, "main", m)
    return seen


def _frame(n):
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[0, 0, 0], f[0, 0, 1] = n % 256, n // 256
    return f


SCENES = [(0, 37), (37, 90), (90, 91), (91, 150), (150, 233)]
STRATS = ["TRACK", "GENERAL", "TRACK", "SPLIT", "TRACK"]
N = 233


def _dense(seen, orig_w=100):
    frames = iter([_frame(n) for n in range(N)])
    cam, tr = _Cameraman(), _Tracker()
    xs = reframe_v2._trajectory_loop(lambda: next(frames, None), False, None,
                                     SCENES, STRATS, orig_w, 1.0, cam, tr)
    return xs, list(seen), cam.__dict__, tr.__dict__


def test_detection_frames_are_the_frames_the_full_pass_detects_on(fake_main):
    _xs, seen, _c, _t = _dense(fake_main)
    detected = sorted({n for kind, n in seen})
    assert reframe_v2._detection_frames(SCENES, STRATS, N, 4, True) == detected


def test_sparse_pass_gives_the_same_trajectory(fake_main):
    xs, seen, cam_state, tracker_state = _dense(fake_main)
    fake_main.clear()
    wanted = iter(reframe_v2._detection_frames(SCENES, STRATS, N, 4, True))

    def next_frame(frame_number):
        assert next(wanted) == frame_number
        return _frame(frame_number)

    cam, tr = _Cameraman(), _Tracker()
    sparse = reframe_v2._trajectory_loop(next_frame, True, N, SCENES, STRATS,
                                         100, 1.0, cam, tr)
    assert sparse == xs
    assert fake_main == seen  # same detector calls, same order
    assert cam.__dict__ == cam_state and tr.__dict__ == tracker_state
    assert next(wanted, None) is None


def test_sparse_trajectory_restores_state_and_falls_back(monkeypatch, fake_main):
    """A sparse pass that does not see the frames it expects must leave the
    camera as it found it and let the full pass run."""
    xs, _seen, _c, _t = _dense(fake_main, orig_w=4)
    monkeypatch.setattr(gpu_decode, "decoded_frames", lambda path: N)
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=False, backend=None: CPU)

    def broken_sparse(*args, **kw):
        cam, tr = args[-2], args[-1]
        cam.target_center_x = 999.0
        tr.resets = 77
        raise reframe_v2._SparseMismatch("stream ended")

    monkeypatch.setattr(reframe_v2, "_sparse_trajectory", broken_sparse)

    class _Proc:
        def __init__(self, *a, **k):
            self.stdout = _Stream()

        def wait(self):
            return 0

    class _Stream:
        def __init__(self):
            self.frames = [_frame(n).tobytes() for n in range(N)]

        def read(self, size):
            return self.frames.pop(0) if self.frames else b""

        def close(self):
            pass

    monkeypatch.setattr(reframe_v2.subprocess, "Popen", _Proc)
    monkeypatch.setattr(reframe_v2, "_analysis_size", lambda w, h: (4, 4))
    cam, tr = _Cameraman(), _Tracker()
    out = reframe_v2._analyze_trajectory("clip.mp4", SCENES, STRATS, 30.0, 4, 4, cam, tr)
    assert out == xs
    assert tr.resets != 77


# --- frame count bookkeeping and the sampler fallback -----------------------

def test_decoded_count_is_tied_to_the_exact_file(tmp_path):
    import os

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 10)
    gpu_decode.remember_count(str(clip), 321)
    assert gpu_decode.decoded_frames(str(clip)) == 321
    clip.write_bytes(b"y" * 11)
    os.utime(clip, ns=(1, 1))
    assert gpu_decode.decoded_frames(str(clip)) is None
    assert gpu_decode.decoded_frames(str(tmp_path / "missing.mp4")) is None


def test_sampler_uses_read_at_when_the_card_is_not_usable(monkeypatch):
    import frame_sampler

    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=False, backend=None: CPU)
    monkeypatch.setattr(frame_sampler, "read_at",
                        lambda cap, idx, max_forward=0: iter([("cpu", i) for i in idx]))
    assert list(frame_sampler.read_at_path("clip.mp4", None, [3, 1])) == [("cpu", 3), ("cpu", 1)]
    assert list(frame_sampler.read_at_path("clip.mp4", None, [1, 3])) == [("cpu", 1), ("cpu", 3)]


def test_empty_sampler_request_does_not_probe_a_decoder(monkeypatch):
    import frame_sampler

    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda *args, **kwargs: pytest.fail("empty request must not probe"))
    assert list(frame_sampler.read_at_path("clip.mp4", None, [])) == []


def test_sampler_read_error_preserves_cpu_remainder_fallback(monkeypatch):
    import frame_sampler

    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=False, backend=None: VIDEOTOOLBOX)
    monkeypatch.setattr(
        frame_sampler, "read_at",
        lambda cap, idx, max_forward=0: iter([("cpu", i) for i in idx]))

    class _Cap:
        def get(self, prop):
            return 2

    class _Stream:
        closed = False

        def readinto(self, _buffer):
            raise OSError("pipe failed")

        def close(self):
            self.closed = True

    class _Proc:
        def __init__(self):
            self.stdout = _Stream()
            self.returncode = None

        def kill(self):
            self.returncode = -9

        def wait(self):
            if self.returncode is None:
                self.returncode = 1
            return self.returncode

    monkeypatch.setattr(frame_sampler.subprocess, "Popen", lambda *a, **k: _Proc())
    assert list(frame_sampler.read_at_path(
        "clip.mp4", _Cap(), [1, 2])) == [("cpu", 1), ("cpu", 2)]


def test_frame_count_is_kept_only_when_the_cfr_sync_changed_nothing():
    plain = b"frame=  810 fps=0.0 q=-0.0 Lsize= 3149kB time=00:00:27.00 speed=6x\n"
    dup = b"frame=  810 fps=0.0 q=-0.0 Lsize= 3149kB time=00:00:27.00 dup=1 drop=0 speed=6x\n"
    assert gpu_decode.one_frame_per_decoded_frame(plain)
    assert not gpu_decode.one_frame_per_decoded_frame(dup)
    assert not gpu_decode.one_frame_per_decoded_frame(b"")
    assert not gpu_decode.one_frame_per_decoded_frame(None)


def test_dense_card_decode_that_fails_is_redone_on_the_cpu(monkeypatch, fake_main):
    """No frame count (irregular timestamps): every frame is decoded, on the
    card first; a card ffmpeg that exits with an error is thrown away."""
    xs, _seen, _c, _t = _dense(fake_main, orig_w=4)
    monkeypatch.setattr(gpu_decode, "decoded_frames", lambda path: None)
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=False, backend=None: (
            CPU if backend == "cpu" else CUDA))
    runs = []

    class _Proc:
        def __init__(self, cmd, **k):
            self.gpu = "-hwaccel" in cmd
            runs.append(self.gpu)
            # The card run dies after 50 frames.
            frames = [_frame(n).tobytes() for n in range(50 if self.gpu else N)]
            self.stdout = types.SimpleNamespace(
                read=lambda size: frames.pop(0) if frames else b"", close=lambda: None)

        def wait(self):
            return 1 if self.gpu else 0

    monkeypatch.setattr(reframe_v2.subprocess, "Popen", _Proc)
    monkeypatch.setattr(reframe_v2, "_analysis_size", lambda w, h: (4, 4))
    cam, tr = _Cameraman(), _Tracker()
    out = reframe_v2._analyze_trajectory("clip.mp4", SCENES, STRATS, 30.0, 4, 4, cam, tr)
    assert runs == [True, False]
    assert out == xs


def test_dense_hardware_start_failure_is_redone_on_the_cpu(monkeypatch, fake_main):
    xs, _seen, _c, _t = _dense(fake_main, orig_w=4)
    monkeypatch.setattr(gpu_decode, "decoded_frames", lambda path: None)
    monkeypatch.setattr(
        gpu_decode, "select_attempt",
        lambda path, render=False, backend=None: (
            CPU if backend == "cpu" else VIDEOTOOLBOX))
    runs = []

    class _Stream:
        def __init__(self):
            self.frames = [_frame(n).tobytes() for n in range(N)]

        def read(self, size):
            return self.frames.pop(0) if self.frames else b""

        def close(self):
            pass

    class _Proc:
        def __init__(self):
            self.stdout = _Stream()

        def wait(self):
            return 0

    def popen(command, **kwargs):
        hardware = "-hwaccel" in command
        runs.append(hardware)
        if hardware:
            raise OSError("VideoToolbox unavailable")
        return _Proc()

    monkeypatch.setattr(reframe_v2.subprocess, "Popen", popen)
    monkeypatch.setattr(reframe_v2, "_analysis_size", lambda w, h: (4, 4))
    cam, tr = _Cameraman(), _Tracker()
    out = reframe_v2._analyze_trajectory(
        "clip.mp4", SCENES, STRATS, 30.0, 4, 4, cam, tr)
    assert runs == [True, False]
    assert out == xs


def test_any_of_stays_shallow_for_many_terms():
    """A flat sum of more than ~100 terms is refused by ffmpeg."""
    terms = [f"eq(n,{i})" for i in range(0, 1500, 3)]
    expr = gpu_decode.any_of(terms)
    depth = deepest = 0
    for ch in expr:
        depth += ch == "("
        depth -= ch == ")"
        deepest = max(deepest, depth)
    assert deepest <= 10
    assert _selected(expr, 1520) == list(range(0, 1500, 3))
    assert gpu_decode.any_of([]) == "0"
    assert gpu_decode.any_of(["eq(n,4)"]) == "eq(n,4)"
