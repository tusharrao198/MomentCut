from types import SimpleNamespace

from platforms.macos import capabilities


def test_snapshot_check_is_local_only(monkeypatch, tmp_path):
    revision = "a" * 40
    calls = {}

    def snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / revision)

    (tmp_path / revision).mkdir()
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert capabilities._snapshot(lambda: ("owner/model", revision))["cached"]
    assert calls == {
        "repo_id": "owner/model",
        "revision": revision,
        "local_files_only": True,
    }


def test_capability_result_does_not_require_videotoolbox(monkeypatch):
    monkeypatch.setattr(capabilities.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(capabilities.accelerator_runtime, "apple_acceleration_state", lambda: "active")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_whisper_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_parakeet_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities, "_module", lambda _name: {"available": True, "version": "test"})
    monkeypatch.setattr(capabilities, "_snapshot", lambda _resolve: {"cached": True})
    monkeypatch.setattr(capabilities, "_ffmpeg_capabilities", lambda **_kwargs: {
        "ffmpeg": {"available": True},
        "ffprobe": {"available": True},
        "videotoolbox": {"encoder_usable": False},
    })

    report = capabilities.collect()

    assert report["ready"] is True
    assert report["ffmpeg"]["videotoolbox"]["encoder_usable"] is False


def test_parakeet_mode_requires_its_cached_model(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setattr(capabilities.accelerator_runtime, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(capabilities.accelerator_runtime, "apple_acceleration_state", lambda: "active")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_whisper_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_parakeet_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities, "_module", lambda _name: {"available": True, "version": "test"})
    checks = iter(({"cached": True}, {"cached": False, "error": "missing"}))
    monkeypatch.setattr(capabilities, "_snapshot", lambda _resolve: next(checks))
    monkeypatch.setattr(capabilities, "_ffmpeg_capabilities", lambda **_kwargs: {
        "ffmpeg": {"available": True},
        "ffprobe": {"available": True},
        "videotoolbox": {"encoder_usable": False},
    })

    report = capabilities.collect()

    assert report["parakeet"]["required"] is True
    assert report["ready"] is False


def test_non_macos_report_is_not_ready(monkeypatch):
    monkeypatch.setattr(capabilities.accelerator_runtime, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(capabilities.accelerator_runtime, "apple_acceleration_state", lambda: "not_apple")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_whisper_engine", lambda: "faster-whisper")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_parakeet_engine", lambda: "onnx")
    monkeypatch.setattr(capabilities, "_module", lambda _name: {"available": False})
    monkeypatch.setattr(capabilities, "_snapshot", lambda _resolve: {"cached": False})
    monkeypatch.setattr(capabilities, "_ffmpeg_capabilities", lambda **_kwargs: {
        "ffmpeg": {"available": True}, "ffprobe": {"available": True},
        "videotoolbox": {"encoder_usable": False},
    })

    assert capabilities.collect()["ready"] is False


def test_runtime_import_is_isolated(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="0.4.3\n", stderr="")

    monkeypatch.setattr(capabilities.subprocess, "run", run)

    assert capabilities._module("mlx_whisper") == {
        "available": True,
        "version": "0.4.3",
    }
    command, kwargs = commands[0]
    assert command[0] == capabilities.sys.executable
    assert command[-1] == "mlx_whisper"
    assert kwargs["timeout"] == 30
    assert kwargs["check"] is False


def test_ffmpeg_skips_hardware_probe_off_mac(monkeypatch):
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: "/test/bin")
    monkeypatch.setattr(
        capabilities.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        capabilities.macos_ffmpeg,
        "_probe_encoder",
        lambda: (_ for _ in ()).throw(AssertionError("should not probe")),
    )

    assert capabilities._ffmpeg_capabilities(apple_silicon=False)["videotoolbox"] == {
        "encoder_listed": False,
        "hwaccel_listed": False,
        "encoder_usable": False,
    }


def test_invalid_gate_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(capabilities.accelerator_runtime, "is_apple_silicon", lambda: True)

    def invalid_gate():
        raise ValueError("bad gate")

    monkeypatch.setattr(
        capabilities.accelerator_runtime, "apple_acceleration_state", invalid_gate)
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_whisper_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities.accelerator_runtime, "resolve_parakeet_engine", lambda: "mlx")
    monkeypatch.setattr(capabilities, "_module", lambda _name: {"available": True})
    monkeypatch.setattr(capabilities, "_snapshot", lambda _resolve: {"cached": True})
    monkeypatch.setattr(capabilities, "_ffmpeg_capabilities", lambda **_kwargs: {
        "ffmpeg": {"available": True},
        "ffprobe": {"available": True},
        "videotoolbox": {},
    })

    report = capabilities.collect()

    assert report["ready"] is False
    assert report["host"]["apple_acceleration"] == "unresolved"
    assert report["host"]["configuration_error"] == "ValueError: bad gate"
