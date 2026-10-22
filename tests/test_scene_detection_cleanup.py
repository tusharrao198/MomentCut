import sys
from types import SimpleNamespace

import scene_detection


def test_transnet_frame_buffer_is_writable(monkeypatch):
    frame_bytes = scene_detection._TN2_W * scene_detection._TN2_H * 3
    completed = SimpleNamespace(
        returncode=0, stdout=bytes(frame_bytes * 2), stderr=b"", args=["ffmpeg"]
    )
    monkeypatch.setattr(
        scene_detection.gpu_decode, "run_with_backend",
        lambda *_args, **_kwargs: completed)
    monkeypatch.setattr(
        scene_detection.gpu_decode, "one_frame_per_decoded_frame",
        lambda _stderr: False)

    frames = scene_detection._extract_frames_small("video.mp4")

    assert frames.shape == (2, scene_detection._TN2_H, scene_detection._TN2_W, 3)
    assert frames.flags.writeable


def test_frame_marker_has_the_scene_consumer_contract():
    marker = scene_detection.frame_marker(90, 30)

    assert marker.get_frames() == 90
    assert marker.get_seconds() == 3.0


def test_non_apple_routes_keep_pyscenedetect_frame_timecodes(monkeypatch):
    calls = []

    class Timecode:
        def __init__(self, frame, fps):
            calls.append((frame, fps))

    monkeypatch.setattr(
        scene_detection.accelerator_runtime,
        "apple_acceleration_active", lambda: False)
    monkeypatch.setitem(sys.modules, "scenedetect", SimpleNamespace(FrameTimecode=Timecode))

    assert isinstance(scene_detection.scene_boundary(90, 30), Timecode)
    assert calls == [(90, 30.0)]


def test_mps_cleanup_uses_mac_adapter_only_when_apple_gate_is_active(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scene_detection.accelerator_runtime,
        "apple_acceleration_active",
        lambda: False,
    )
    assert scene_detection._clear_mps_cache_if_active("mps") is False

    monkeypatch.setattr(
        scene_detection.accelerator_runtime,
        "apple_acceleration_active",
        lambda: True,
    )
    from platforms.macos import cleanup
    monkeypatch.setattr(
        cleanup, "clear_mps_cache", lambda device: calls.append(device) or True)

    assert scene_detection._clear_mps_cache_if_active("mps") is True
    assert calls == ["mps"]


def test_releasing_scene_model_drops_the_reference_before_cleanup(monkeypatch):
    model = SimpleNamespace(device="mps")
    calls = []
    monkeypatch.setattr(scene_detection, "_tn2_model", model)
    monkeypatch.setattr(
        scene_detection, "_clear_mps_cache_if_active",
        lambda device: calls.append((scene_detection._tn2_model, device)) or True)

    assert scene_detection.release_transnetv2_model() is True
    assert scene_detection._tn2_model is None
    assert calls == [(None, "mps")]
