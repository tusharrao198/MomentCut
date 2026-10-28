from dataclasses import replace
import signal
import subprocess

import pytest

import audio_events
from platforms.macos import audio_events as macos_audio_events


def _settings(model, worker):
    return audio_events.AudioEventSettings.from_env({
        "AUDIO_EVENTS_MODE": "ced",
        "AUDIO_EVENTS_MODEL_PATH": str(model),
        "AUDIO_EVENTS_CED_WORKER_PATH": str(worker),
    })


def _worker(path, *, metal=True, observed="MTL0", requested="metal"):
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        f"  printf '%s\\n' '{{\"protocol_version\":1,\"worker_build\":\"ced-test\",\"metal_compiled\":{str(metal).lower()}}}'\n"
        "elif [ \"$1\" = \"--probe\" ]; then\n"
        f"  printf '%s\\n' '{{\"protocol_version\":1,\"worker_build\":\"ced-test\",\"model_sha256\":\"{audio_events._sha256_file(str(path.parent / 'model.gguf'))}\",\"runtime\":{{\"requested_device\":\"{requested}\",\"observed_device\":\"{observed}\"}}}}'\n"
        "else\n"
        "  exit 64\n"
        "fi\n"
    )
    path.chmod(0o700)


def test_resolve_settings_requires_regular_metal_worker(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker)
    monkeypatch.setattr(macos_audio_events.accelerator_runtime, "require_apple_acceleration", lambda _name: None)

    resolved = macos_audio_events.resolve_settings(_settings(model, worker))

    assert resolved.model_sha256 == audio_events._sha256_file(str(model))
    assert resolved.worker_build == "ced-test"


def test_static_acceptance_resolution_does_not_load_ced(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker)
    monkeypatch.setattr(
        macos_audio_events.accelerator_runtime,
        "require_apple_acceleration",
        lambda _name: None,
    )
    monkeypatch.setattr(
        macos_audio_events,
        "_probe",
        lambda _settings: pytest.fail("acceptance must not load the CED model"),
    )

    resolved = macos_audio_events.resolve_settings(
        _settings(model, worker), verify_execution=False,
    )

    assert resolved.worker_build == "ced-test"


def test_resolve_settings_rejects_non_metal_worker(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker, metal=False)
    monkeypatch.setattr(macos_audio_events.accelerator_runtime, "require_apple_acceleration", lambda _name: None)

    with pytest.raises(macos_audio_events.MacAudioEventsError, match="Metal"):
        macos_audio_events.resolve_settings(_settings(model, worker))


def test_worker_environment_is_narrow_and_selects_requested_device(monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("GEMINI_API_KEY", "not-forwarded")
    base = audio_events.AudioEventSettings.from_env({"AUDIO_EVENTS_MODE": "disabled"})

    assert macos_audio_events._worker_env(replace(
        base, mode="ced", device="metal",
    ))["CED_DEVICE"] == "MTL0"
    assert macos_audio_events._worker_env(replace(
        base, mode="ced", device="cpu", allow_cpu_fallback=True,
    ))["CED_DEVICE"] == "cpu"
    assert "GEMINI_API_KEY" not in macos_audio_events._worker_env(replace(
        base, mode="ced", device="metal",
    ))


def test_resolve_settings_rejects_cpu_probe(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker, observed="cpu")
    monkeypatch.setattr(macos_audio_events.accelerator_runtime, "require_apple_acceleration", lambda _name: None)

    with pytest.raises(macos_audio_events.MacAudioEventsError, match="required Metal"):
        macos_audio_events.resolve_settings(_settings(model, worker))


def test_resolve_settings_accepts_explicit_cpu_diagnostic(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker, observed="CPU", requested="cpu")
    monkeypatch.setattr(macos_audio_events.accelerator_runtime, "require_apple_acceleration", lambda _name: None)
    settings = replace(
        _settings(model, worker), device="cpu", require_accelerator=False,
        allow_cpu_fallback=True,
    )

    assert macos_audio_events.resolve_settings(settings).device == "cpu"


def test_run_rejects_out_of_timeline_stream_before_worker_or_extraction(monkeypatch, tmp_path):
    settings = audio_events.AudioEventSettings.from_env({"AUDIO_EVENTS_MODE": "disabled"})
    settings = replace(settings, mode="ced", device="cpu", allow_cpu_fallback=True)
    monkeypatch.setattr(
        macos_audio_events,
        "resolve_worker_for_new_run",
        lambda _settings: pytest.fail("worker resolution must not run"),
    )
    monkeypatch.setattr(
        macos_audio_events.audio_events,
        "extract_pcm_wav",
        lambda *_args: pytest.fail("PCM extraction must not run"),
    )

    with pytest.raises(audio_events.AudioEventError, match="outside"):
        macos_audio_events.run(
            settings,
            source_file=str(tmp_path / "source.mp4"),
            source_duration_seconds=10.0,
            audio_stream={"start_seconds": 10.0},
            project_dir=str(tmp_path),
        )


def test_run_caps_windows_from_decoded_audio_not_full_video(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker)
    settings = replace(
        _settings(model, worker),
        model_sha256=audio_events._sha256_file(str(model)),
        worker_build="ced-test",
        max_windows=3,
    )

    class Process:
        pid = 125
        returncode = 1

        def communicate(self, timeout=None):
            return "", "controlled worker failure"

    launched = []
    monkeypatch.setattr(macos_audio_events, "resolve_worker_for_new_run", lambda _settings: None)
    monkeypatch.setattr(
        macos_audio_events.audio_events,
        "extract_pcm_wav",
        lambda *_args: 5.0,
    )
    monkeypatch.setattr(
        macos_audio_events.subprocess,
        "Popen",
        lambda *args, **kwargs: launched.append((args, kwargs)) or Process(),
    )

    with pytest.raises(macos_audio_events.MacAudioEventsError, match="controlled worker failure"):
        macos_audio_events.run(
            settings,
            source_file=str(model),
            source_duration_seconds=3600.0,
            audio_stream={"start_seconds": 0.0},
            project_dir=str(tmp_path),
        )

    assert len(launched) == 1


def test_run_kills_an_unresponsive_worker_after_timeout(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker)
    settings = replace(
        _settings(model, worker),
        model_sha256=audio_events._sha256_file(str(model)),
        worker_build="ced-test",
        timeout_seconds=1,
    )

    class Process:
        pid = 123
        returncode = None
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls < 3:
                raise subprocess.TimeoutExpired("ced-worker", timeout)
            return "", ""

    process = Process()
    signals = []
    monkeypatch.setattr(macos_audio_events, "resolve_worker_for_new_run", lambda _settings: None)
    monkeypatch.setattr(
        macos_audio_events.audio_events,
        "extract_pcm_wav",
        lambda *_args: 10.0,
    )
    monkeypatch.setattr(macos_audio_events.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        macos_audio_events.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    with pytest.raises(macos_audio_events.MacAudioEventsError, match="timed out"):
        macos_audio_events.run(
            settings,
            source_file=str(model),
            source_duration_seconds=10.0,
            audio_stream={"start_seconds": 0.0},
            project_dir=str(tmp_path),
        )

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_run_stops_worker_when_parent_is_interrupted(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    _worker(worker)
    settings = replace(
        _settings(model, worker),
        model_sha256=audio_events._sha256_file(str(model)),
        worker_build="ced-test",
    )

    class Process:
        pid = 124
        returncode = None
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

    process = Process()
    signals = []
    monkeypatch.setattr(macos_audio_events, "resolve_worker_for_new_run", lambda _settings: None)
    monkeypatch.setattr(
        macos_audio_events.audio_events,
        "extract_pcm_wav",
        lambda *_args: 10.0,
    )
    monkeypatch.setattr(macos_audio_events.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        macos_audio_events.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    with pytest.raises(KeyboardInterrupt):
        macos_audio_events.run(
            settings,
            source_file=str(model),
            source_duration_seconds=10.0,
            audio_stream={"start_seconds": 0.0},
            project_dir=str(tmp_path),
        )

    assert signals == [(124, signal.SIGTERM)]


def test_resolve_settings_preserves_probe_worker_error(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    worker = tmp_path / "worker"
    worker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf '%s\\n' '{\"protocol_version\":1,\"worker_build\":\"ced-test\",\"metal_compiled\":true}'\n"
        "else\n"
        "  printf '%s\\n' 'Metal command queue unavailable' >&2\n"
        "  exit 70\n"
        "fi\n"
    )
    worker.chmod(0o700)
    monkeypatch.setattr(macos_audio_events.accelerator_runtime, "require_apple_acceleration", lambda _name: None)

    with pytest.raises(macos_audio_events.MacAudioEventsError, match="Metal command queue unavailable"):
        macos_audio_events.resolve_settings(_settings(model, worker))
