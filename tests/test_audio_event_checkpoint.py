from dataclasses import replace

import audio_events
import checkpoint_store
import job_state


def _settings():
    initial = audio_events.AudioEventSettings.from_env({
        "AUDIO_EVENTS_MODE": "ced",
        "AUDIO_EVENTS_MODEL_PATH": "/models/ced.gguf",
        "AUDIO_EVENTS_CED_WORKER_PATH": "/bin/ced-worker",
    })
    return replace(initial, model_sha256="a" * 64, worker_build="ced.cpp-test")


def _config(settings):
    return job_state.seal_job_config({
        "schema_version": 1,
        "job_id": "ced-job",
        "configuration_revision": 1,
        "project_dir": "/tmp/ced-job",
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "asr": {"requested_backend": "whisper", "whisper": {"model_revision": None}},
        "selection": {"clip_min_seconds": 15, "clip_max_seconds": 60},
        "render": {},
        "audio_events": settings.durable_config(),
        "environment": {},
        "credential_requirements": {},
        "checkpointing_enabled": True,
    })


def _identity():
    return {"sha256": "b" * 64, "size": 7, "duration_ms": 60_000, "fps": 30.0}


def _artifact(settings):
    return {
        "schema_version": 1,
        "provider": "ced",
        "profile": {"id": "gameplay-v1", "version": 1, "sha256": settings.profile_sha256},
        "identity": {"source_sha256": "b" * 64, "model_sha256": "a" * 64, "worker_build": "ced.cpp-test", "sample_rate_hz": 16000},
        "runtime": {"requested_device": "metal", "observed_device": "MTL0"},
        "audio_stream": {"absolute_stream_index": 1, "start_seconds": 0.0, "selection_rule": "default"},
        "events": [{"label": "Explosion", "start_seconds": 2.0, "end_seconds": 12.0, "max_score": 0.9}],
        "summary": {"qualified_event_count": 1, "window_count": 26, "gpu_residency_note": "MTL0 confirms native backend selection, not per-operation residency."},
    }


def test_sparse_asr_plus_valid_event_checkpoint_allows_speech_plan(tmp_path):
    settings = _settings()
    config = _config(settings)
    config["project_dir"] = str(tmp_path)
    config = job_state.seal_job_config({key: value for key, value in config.items() if key != "job_config_hash"})
    identity = _identity()
    artifact = _artifact(settings)
    artifact_file = tmp_path / "audio-events.json"
    job_state.atomic_write_json(str(artifact_file), artifact)
    descriptor = {"relative_path": artifact_file.name, "size": artifact_file.stat().st_size, "sha256": checkpoint_store.sha256_file(str(artifact_file))}
    store = checkpoint_store.CheckpointStore(str(tmp_path), config, True, 1, identity)
    asr_route = "whisper-mlx"
    asr_key = checkpoint_store.expected_asr_key(config, identity, asr_route)
    provenance = {"route_context": asr_route, "backend": {"effective_route": asr_route, "effective_backend": "whisper", "immutable_revision_available": False, "immutable_model_revision": None}}
    transcript = {"language": "en", "segments": [{"start": 0.0, "end": 5.0, "text": "hello", "words": []}]}
    assert store.save_fixed("asr", asr_key, {"outcome": "sparse_speech", "transcript": transcript, "word_count": 1}, provenance)
    event_key = checkpoint_store.expected_audio_events_key(config, identity, asr_route)
    assert store.save_fixed("audio-events", event_key, {"artifact": descriptor, "qualified_event_count": 1}, {"route_context": "ced-metal"})
    route = {"selection_route": "speech-local", "asr_effective_route": asr_route, "asr_outcome": "sparse_speech", "audio_events_effective_route": "ced-metal", "audio_events_compatibility_key": event_key, "audio_events_has_qualified_events": True, "selection_model_revision": None}
    plan_key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {"shorts": [{"start": 0.0, "end": 15.0}], "transcript": transcript, "source_analysis": {"source_subtitle_band": None, "layout_config": [], "checkpoints": {}}}
    assert store.save_fixed("clip-plan", plan_key, payload, {"route_context": route})
    assert store.load_current_clip_plan(lambda value: checkpoint_store.expected_clip_plan_key(config, identity, value)) is not None
    assert store.validate_json_artifact(
        {**descriptor, "relative_path": "../audio-events.json"},
        maximum_bytes=settings.max_raw_bytes,
    ) is None


def test_sparse_supplied_transcript_without_source_audio_is_a_text_plan(tmp_path):
    settings = _settings()
    config = _config(settings)
    config["project_dir"] = str(tmp_path)
    config = job_state.seal_job_config({
        key: value for key, value in config.items() if key != "job_config_hash"
    })
    identity = _identity()
    store = checkpoint_store.CheckpointStore(str(tmp_path), config, True, 1, identity)
    transcript = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 15.0, "text": "hello", "words": []}],
    }
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "precomputed-transcript",
        "asr_outcome": "sparse_speech",
        "selection_model_revision": None,
    }
    plan_key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {
        "shorts": [{"start": 0.0, "end": 15.0}],
        "transcript": transcript,
        "source_analysis": {
            "source_subtitle_band": None,
            "layout_config": [],
            "checkpoints": {},
        },
    }
    assert store.save_fixed("clip-plan", plan_key, payload, {"route_context": route})
    assert store.load_current_clip_plan(
        lambda value: checkpoint_store.expected_clip_plan_key(config, identity, value)
    ) is not None


def test_enabled_ced_configuration_changes_visual_plan_identity():
    settings = _settings()
    config = _config(settings)
    identity = _identity()
    route = {
        "selection_route": "silent-local",
        "asr_effective_route": "whisper-mlx",
        "asr_outcome": "sparse_speech",
        "selection_model_revision": "vision-digest",
    }
    original_key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    changed = {key: value for key, value in config.items() if key != "job_config_hash"}
    changed["audio_events"] = dict(changed["audio_events"])
    changed["audio_events"]["window_seconds"] = 12.0
    changed = job_state.seal_job_config(changed)
    assert checkpoint_store.expected_clip_plan_key(changed, identity, route) != original_key
