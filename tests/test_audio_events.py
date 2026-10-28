from dataclasses import replace
from types import SimpleNamespace

import pytest

import app
import audio_events


def _settings():
    configured = audio_events.AudioEventSettings.from_env({
        "AUDIO_EVENTS_MODE": "ced",
        "AUDIO_EVENTS_MODEL_PATH": "/models/ced.gguf",
        "AUDIO_EVENTS_CED_WORKER_PATH": "/bin/ced-worker",
    })
    return replace(
        configured,
        model_sha256="a" * 64,
        worker_build="ced.cpp-test",
    )


def _raw(settings, *, stream_start=-2.0, starts=None):
    labels = [
        "Gunshot, gunfire",
        "Machine gun",
        "Fusillade",
        "Artillery fire",
        "Explosion",
        "Boom",
        "Music",
    ]
    if starts is None:
        starts = tuple(
            stream_start + index * settings.hop_seconds
            for index in range(audio_events.window_count(
                20.0, settings, stream_start_seconds=stream_start,
            ))
        )
    windows = []
    for index, start in enumerate(starts):
        end = min(start + settings.window_seconds, 20.0)
        windows.append({
            "start_seconds": start,
            "end_seconds": end,
            "padded_tail_seconds": settings.window_seconds - (end - start),
            "scores": [
                0.8 if index == 0 else 0.0,
                0.0,
                0.0,
                0.0,
                0.9 if index else 0.0,
                0.0,
                0.2,
            ],
        })
    return {
        "schema_version": 1,
        "provider": "ced",
        "worker_build": settings.worker_build,
        "model_sha256": settings.model_sha256,
        "runtime": {"requested_device": "metal", "observed_device": "MTL0"},
        "audio": {
            "sample_rate_hz": 16000,
            "stream_start_seconds": stream_start,
            "window_seconds": 10.0,
            "hop_seconds": 2.0,
            "decoded_duration_seconds": 20.0 - stream_start,
            "window_count": len(windows),
        },
        "labels": labels,
        "windows": windows,
    }


def _source():
    return {"sha256": "b" * 64, "size": 123, "duration_ms": 20_000}


def test_disabled_settings_are_inert():
    settings = audio_events.AudioEventSettings.from_env({"AUDIO_EVENTS_MODE": "disabled"})

    assert settings.enabled is False
    assert settings.durable_config() == {"enabled": False}
    assert audio_events.valid_durable_config(settings.durable_config())


def test_invalid_enabled_settings_fail_before_native_imports():
    with pytest.raises(audio_events.AudioEventError, match="MODEL_PATH"):
        audio_events.AudioEventSettings.from_env({"AUDIO_EVENTS_MODE": "ced"})
    with pytest.raises(audio_events.AudioEventError, match="opposite"):
        audio_events.AudioEventSettings.from_env({
            "AUDIO_EVENTS_MODE": "ced",
            "AUDIO_EVENTS_MODEL_PATH": "/model",
            "AUDIO_EVENTS_CED_WORKER_PATH": "/worker",
            "AUDIO_EVENTS_REQUIRE_ACCELERATOR": "true",
            "AUDIO_EVENTS_ALLOW_CPU_FALLBACK": "true",
        })
    with pytest.raises(audio_events.AudioEventError, match="CPU fallback"):
        audio_events.AudioEventSettings.from_env({
            "AUDIO_EVENTS_MODE": "ced",
            "AUDIO_EVENTS_DEVICE": "cpu",
            "AUDIO_EVENTS_MODEL_PATH": "/model",
            "AUDIO_EVENTS_CED_WORKER_PATH": "/worker",
            "AUDIO_EVENTS_REQUIRE_ACCELERATOR": "true",
            "AUDIO_EVENTS_ALLOW_CPU_FALLBACK": "false",
        })


def test_normalization_clamps_negative_stream_time_and_retains_events():
    settings = _settings()
    artifact = audio_events.normalize_ced_output(
        _raw(settings), settings, source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
    )

    assert artifact["summary"]["qualified_event_count"] == 2
    gunfire = next(event for event in artifact["events"] if event["label"] == "Gunshot, gunfire")
    assert gunfire["start_seconds"] == 0.0
    assert audio_events.validate_event_artifact(
        artifact, settings=settings, source_identity=_source(),
    )


def test_normalization_rejects_worker_timing_drift():
    settings = _settings()
    raw = _raw(settings)
    raw["windows"][1]["start_seconds"] = 0.5
    with pytest.raises(audio_events.AudioEventError, match="window position"):
        audio_events.normalize_ced_output(
            raw, settings, source_identity=_source(), source_duration_seconds=20.0,
            audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
        )


def test_normalization_rejects_incomplete_window_sequence():
    settings = _settings()
    raw = _raw(settings)
    raw["windows"].pop()
    raw["audio"]["window_count"] -= 1
    with pytest.raises(audio_events.AudioEventError, match="window count"):
        audio_events.normalize_ced_output(
            raw, settings, source_identity=_source(), source_duration_seconds=20.0,
            audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
        )


def test_normalization_accepts_audio_stream_shorter_than_video():
    settings = _settings()
    raw = _raw(settings, stream_start=0.0, starts=(0.0, 2.0, 4.0))
    raw["audio"]["decoded_duration_seconds"] = 5.0
    for index, window in enumerate(raw["windows"]):
        window["end_seconds"] = 5.0
        window["padded_tail_seconds"] = 5.0 + index * 2.0

    artifact = audio_events.normalize_ced_output(
        raw, settings, source_identity=_source(), source_duration_seconds=20.0,
        audio_stream={"absolute_stream_index": 3, "start_seconds": 0.0, "selection_rule": "default"},
    )

    assert artifact["summary"]["window_count"] == 3


def test_normalization_rejects_audio_duration_mismatching_last_window():
    settings = _settings()
    raw = _raw(settings, stream_start=0.0, starts=(0.0, 2.0, 4.0))
    raw["audio"]["decoded_duration_seconds"] = 5.0

    with pytest.raises(audio_events.AudioEventError, match="decoded audio duration"):
        audio_events.normalize_ced_output(
            raw, settings, source_identity=_source(), source_duration_seconds=20.0,
            audio_stream={"absolute_stream_index": 3, "start_seconds": 0.0, "selection_rule": "default"},
        )


def test_normalization_rejects_missing_profile_label():
    settings = _settings()
    raw = _raw(settings)
    raw["labels"].remove("Boom")
    for window in raw["windows"]:
        window["scores"].pop(5)
    with pytest.raises(audio_events.AudioEventError, match="required gameplay"):
        audio_events.normalize_ced_output(
            raw, settings, source_identity=_source(), source_duration_seconds=20.0,
            audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
        )


def test_event_windows_are_additive_and_disable_word_snapping():
    settings = _settings()
    artifact = audio_events.normalize_ced_output(
        _raw(settings), settings, source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
    )
    windows = audio_events.build_selection_windows(
        [{"id": "speech_001", "start": 0.0, "end": 15.0, "text": "hello"}],
        artifact, source_duration_seconds=20.0, maximum_windows=2,
        minimum_window_seconds=15.0,
    )

    assert windows[0]["word_snapping_allowed"] is True
    assert windows[0]["audio_events"]
    event_window = next(window for window in windows if window["id"].startswith("audio_event_"))
    assert event_window["word_snapping_allowed"] is False
    assert event_window["text"].startswith("[No speech transcript")
    assert event_window["end"] - event_window["start"] >= 15.0


def test_artifact_rejects_unqualified_or_unknown_event_label():
    settings = _settings()
    artifact = audio_events.normalize_ced_output(
        _raw(settings), settings, source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
    )
    artifact["events"][0]["label"] = "Unknown event"
    assert not audio_events.validate_event_artifact(artifact)


def test_artifact_reuse_rechecks_the_sealed_device_policy():
    settings = _settings()
    artifact = audio_events.normalize_ced_output(
        _raw(settings), settings, source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={"absolute_stream_index": 3, "start_seconds": -2.0, "selection_rule": "default"},
    )

    assert audio_events.validate_event_artifact(artifact, settings=settings)
    artifact["runtime"] = {"requested_device": "cpu", "observed_device": "cpu"}
    assert not audio_events.validate_event_artifact(artifact, settings=settings)


def test_effective_route_matches_the_sealed_provider_and_device():
    settings = _settings()

    assert audio_events.effective_route(settings) == "ced-metal"
    assert audio_events.effective_route(replace(
        settings, device="cpu", require_accelerator=False, allow_cpu_fallback=True,
    )) == "ced-cpu"
    with pytest.raises(audio_events.AudioEventError, match="no effective route"):
        audio_events.effective_route(
            audio_events.AudioEventSettings.from_env({"AUDIO_EVENTS_MODE": "disabled"})
        )


def test_source_audio_duration_rejects_stream_outside_source_timeline():
    assert audio_events.source_audio_duration(10.0, -2.0) == 12.0
    with pytest.raises(audio_events.AudioEventError, match="outside"):
        audio_events.source_audio_duration(10.0, 10.0)


def test_enabled_settings_require_exact_pcm_window_boundaries():
    env = {
        "AUDIO_EVENTS_MODE": "ced",
        "AUDIO_EVENTS_MODEL_PATH": "/model",
        "AUDIO_EVENTS_CED_WORKER_PATH": "/worker",
        "AUDIO_EVENTS_HOP_SECONDS": "0.0001",
    }

    with pytest.raises(audio_events.AudioEventError, match="exact multiple"):
        audio_events.AudioEventSettings.from_env(env)


def test_enabled_settings_bound_compute_and_candidate_caps():
    base = {
        "AUDIO_EVENTS_MODE": "ced",
        "AUDIO_EVENTS_MODEL_PATH": "/model",
        "AUDIO_EVENTS_CED_WORKER_PATH": "/worker",
    }
    with pytest.raises(audio_events.AudioEventError, match="MAX_WINDOWS cannot exceed"):
        audio_events.AudioEventSettings.from_env({
            **base, "AUDIO_EVENTS_MAX_WINDOWS": "10001",
        })
    with pytest.raises(
        audio_events.AudioEventError,
        match="MAX_CANDIDATE_WINDOWS cannot exceed",
    ):
        audio_events.AudioEventSettings.from_env({
            **base, "AUDIO_EVENTS_MAX_CANDIDATE_WINDOWS": "10001",
        })

    durable = _settings().durable_config()
    durable["max_windows"] = 10_001
    assert not audio_events.valid_durable_config(durable)


def test_durable_config_and_worker_json_reject_boolean_numbers():
    durable = _settings().durable_config()
    durable["max_windows"] = True
    assert not audio_events.valid_durable_config(durable)
    durable = _settings().durable_config()
    durable["profile_version"] = True
    assert not audio_events.valid_durable_config(durable)

    raw = _raw(_settings())
    raw["schema_version"] = True
    with pytest.raises(audio_events.AudioEventError, match="unsupported schema"):
        audio_events.normalize_ced_output(
            raw,
            _settings(),
            source_identity=_source(),
            source_duration_seconds=20.0,
            audio_stream={
                "absolute_stream_index": 3,
                "start_seconds": -2.0,
                "selection_rule": "default",
            },
        )

    raw = _raw(_settings())
    raw["windows"][0]["scores"][0] = True
    with pytest.raises(audio_events.AudioEventError, match="invalid score"):
        audio_events.normalize_ced_output(
            raw,
            _settings(),
            source_identity=_source(),
            source_duration_seconds=20.0,
            audio_stream={
                "absolute_stream_index": 3,
                "start_seconds": -2.0,
                "selection_rule": "default",
            },
        )

    artifact = audio_events.normalize_ced_output(
        _raw(_settings()),
        _settings(),
        source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={
            "absolute_stream_index": 3,
            "start_seconds": -2.0,
            "selection_rule": "default",
        },
    )
    artifact["profile"]["version"] = True
    assert not audio_events.validate_event_artifact(artifact)


def test_window_cap_and_runtime_kill_switch_are_explicit():
    settings = replace(_settings(), max_windows=1)
    with pytest.raises(audio_events.AudioEventError, match="MAX_WINDOWS"):
        audio_events.window_count(20.0, settings)
    with pytest.raises(audio_events.AudioEventError, match="MAX_WINDOWS"):
        audio_events.window_count(20.0, settings, stream_start_seconds=-2.0)
    assert audio_events.runtime_enabled({"AUDIO_EVENTS_RUNTIME_ENABLED": "false"}) is False
    with pytest.raises(audio_events.AudioEventError, match="true or false"):
        audio_events.runtime_enabled({"AUDIO_EVENTS_RUNTIME_ENABLED": "maybe"})


def test_probe_prefers_default_audio_stream(monkeypatch):
    payload = '{"streams":[{"codec_type":"audio","index":1,"start_time":"0"},{"codec_type":"audio","index":4,"start_time":"-0.25","disposition":{"default":1}}]}'
    monkeypatch.setattr(audio_events.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=payload, stderr=""))

    stream = audio_events.probe_audio_stream("ffprobe", "source.mp4")

    assert stream == {
        "absolute_stream_index": 4,
        "start_seconds": -0.25,
        "selection_rule": "default",
        "language": None,
        "title": None,
    }


def test_probe_rejects_a_malformed_stream_list(monkeypatch):
    monkeypatch.setattr(
        audio_events.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout='{"streams":{}}', stderr=""
        ),
    )

    with pytest.raises(audio_events.AudioEventError, match="stream list"):
        audio_events.probe_audio_stream("ffprobe", "source.mp4")


def test_artifact_bound_source_identity_rejects_out_of_timeline_events():
    settings = _settings()
    artifact = audio_events.normalize_ced_output(
        _raw(settings), settings, source_identity=_source(),
        source_duration_seconds=20.0,
        audio_stream={
            "absolute_stream_index": 3,
            "start_seconds": -2.0,
            "selection_rule": "default",
        },
    )
    artifact["events"][0]["end_seconds"] = 20.1

    assert not audio_events.validate_event_artifact(
        artifact, settings=settings, source_identity=_source(),
    )


def test_child_environment_uses_the_current_ced_runtime_switch(monkeypatch):
    monkeypatch.setenv("AUDIO_EVENTS_RUNTIME_ENABLED", "false")

    environment = app.build_job_child_env({"environment": {}}, {})

    assert environment["AUDIO_EVENTS_RUNTIME_ENABLED"] == "false"


def test_enabled_ced_is_rejected_at_acceptance_when_runtime_switch_is_off(monkeypatch):
    monkeypatch.setattr(app, "MAX_CONCURRENT_JOBS", 1)
    with pytest.raises(ValueError, match="RUNTIME_ENABLED"):
        app._audio_events_config({
            "AUDIO_EVENTS_MODE": "ced",
            "AUDIO_EVENTS_MODEL_PATH": "/models/ced.gguf",
            "AUDIO_EVENTS_CED_WORKER_PATH": "/bin/ced-worker",
            "AUDIO_EVENTS_RUNTIME_ENABLED": "false",
        }, checkpointing_enabled=True)
