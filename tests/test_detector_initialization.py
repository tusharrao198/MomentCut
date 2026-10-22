"""Detector setup must not make importing the worker start native models."""
from types import SimpleNamespace

import numpy as np

import main


def test_non_apple_ytdlp_options_preserve_portable_defaults(monkeypatch):
    monkeypatch.setattr(
        main.accelerator_runtime, "apple_acceleration_active", lambda: False)

    assert main._macos_ytdlp_options() == {}


def test_apple_ytdlp_options_use_the_isolated_adapter(monkeypatch):
    from platforms.macos import ytdlp

    monkeypatch.setattr(
        main.accelerator_runtime, "apple_acceleration_active", lambda: True)
    monkeypatch.setattr(
        ytdlp, "javascript_runtime_options",
        lambda: {"js_runtimes": {"node": {"path": "/test/node"}}},
    )

    assert main._macos_ytdlp_options() == {
        "js_runtimes": {"node": {"path": "/test/node"}}
    }


def test_detectors_are_constructed_only_on_first_use(monkeypatch):
    calls = []

    def fake_yolo(path):
        calls.append(("yolo", path))
        return "yolo"

    def fake_face_detection(**kwargs):
        calls.append(("face", kwargs))
        return "face"

    monkeypatch.setattr(main, "model", None)
    monkeypatch.setattr(main, "face_detection", None)
    monkeypatch.setattr(main, "YOLO", fake_yolo)
    monkeypatch.setattr(
        main, "mp_face_detection",
        SimpleNamespace(FaceDetection=fake_face_detection))

    assert calls == []
    assert main._yolo_model() == "yolo"
    assert main._face_detector() == "face"
    assert main._yolo_model() == "yolo"
    assert main._face_detector() == "face"
    assert calls == [
        ("yolo", "yolov8n.pt"),
        ("face", {"model_selection": 1, "min_detection_confidence": 0.5}),
    ]


def test_detection_calls_use_the_lazily_constructed_models(monkeypatch):
    calls = []

    class Face:
        def process(self, frame):
            calls.append(("face", frame.shape))
            return SimpleNamespace(detections=[])

    class Yolo:
        def __call__(self, frame, **kwargs):
            calls.append(("yolo", frame.shape, kwargs))
            return []

    monkeypatch.setattr(main, "face_detection", Face())
    monkeypatch.setattr(main, "model", Yolo())
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    assert main.detect_face_candidates(frame) == []
    assert main.detect_person_yolo(frame) is None
    assert calls == [
        ("face", (100, 200, 3)),
        ("yolo", (100, 200, 3), {"verbose": False, "classes": [0]}),
    ]
