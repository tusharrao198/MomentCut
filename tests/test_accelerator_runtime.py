import pytest

import accelerator_runtime as runtime


def test_auto_selects_mlx_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: True)
    assert runtime.resolve_whisper_engine("auto") == "mlx"

    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: False)
    assert runtime.resolve_whisper_engine("auto") == "faster-whisper"


def test_explicit_faster_whisper_is_platform_independent(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: True)
    assert runtime.resolve_whisper_engine("faster-whisper") == "faster-whisper"


def test_explicit_mlx_rejects_non_apple_host(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: False)
    with pytest.raises(runtime.AcceleratorConfigurationError):
        runtime.resolve_whisper_engine("mlx")


def test_invalid_whisper_engine_is_rejected():
    with pytest.raises(runtime.AcceleratorConfigurationError):
        runtime.resolve_whisper_engine("cuda")


def test_parakeet_auto_selects_mlx_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: True)
    assert runtime.resolve_parakeet_engine("auto") == "mlx"

    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: False)
    assert runtime.resolve_parakeet_engine("auto") == "onnx"


def test_explicit_onnx_is_platform_independent(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: True)
    assert runtime.resolve_parakeet_engine("onnx") == "onnx"


def test_explicit_parakeet_mlx_rejects_non_apple_host(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: False)
    with pytest.raises(runtime.AcceleratorConfigurationError):
        runtime.resolve_parakeet_engine("mlx")
