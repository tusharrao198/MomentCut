from types import SimpleNamespace

import scene_detection


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
