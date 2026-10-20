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


def test_memory_auto_selects_macos_without_probing_nvidia(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: True)
    monkeypatch.setattr(
        runtime.shutil, "which",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("Apple auto must not probe nvidia-smi")))
    assert runtime.resolve_memory_admission_backend("auto") == "macos"


def test_memory_auto_selects_nvidia_then_count(monkeypatch):
    monkeypatch.setattr(runtime, "is_apple_silicon", lambda **_kwargs: False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    assert runtime.resolve_memory_admission_backend("auto") == "nvidia"

    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)
    assert runtime.resolve_memory_admission_backend("auto") == "count"


def test_invalid_memory_backend_is_rejected():
    with pytest.raises(runtime.AcceleratorConfigurationError):
        runtime.resolve_memory_admission_backend("mps")


def test_host_available_memory_is_read_lazily(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
            available=4096 * 1024 * 1024)),
    )
    assert runtime.host_available_mb() == 4096


def test_nvidia_reader_keeps_the_existing_gpu_zero_query(monkeypatch):
    from types import SimpleNamespace

    calls = {}

    def run(command, **kwargs):
        calls.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout="12288\n")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.nvidia_free_mb() == 12288
    assert calls == {
        "command": [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        "capture_output": True,
        "text": True,
        "timeout": 5,
    }
