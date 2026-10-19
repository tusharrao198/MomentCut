import json
import math
import os

import pytest

import checkpoint_store
import job_state


def _config(job_id="job-1", enabled=True):
    return {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "source": {
            "kind": "upload",
            "retained_relative_path": "source.mp4",
            "cap_minutes": None,
        },
        "pipeline": {"output_format": "auto"},
        "asr": {
            "requested_backend": "whisper",
            "whisper": {"model_revision": "whisper-rev-1"},
        },
        "selection": {"target_min": 1, "target_max": 3},
        "render": {"encoder": "libx264"},
        "checkpointing_enabled": enabled,
    }


def _source(data=b"source", duration_ms=1000):
    return {
        "sha256": checkpoint_store.sha256_bytes(data),
        "size": len(data),
        "duration_ms": duration_ms,
        "fps": 30.0,
    }


def _speech():
    return {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "hello", "words": []},
        ],
    }


def _asr_provenance(route="no-audio"):
    backend = {
        "effective_route": route,
        "effective_backend": "none" if route == "no-audio" else "whisper",
    }
    if route != "no-audio":
        backend.update({
            "immutable_revision_available": True,
            "immutable_model_revision": "whisper-rev-1",
        })
    return {"route_context": route, "backend": backend}


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), (" true ", True), ("TRUE", True), ("false", False)],
)
def test_enablement_parser(value, expected):
    env = {} if value is None else {"PIPELINE_CHECKPOINTS_ENABLED": value}
    assert checkpoint_store.pipeline_checkpoints_enabled(env) is expected


def test_invalid_enablement_fails_clearly():
    with pytest.raises(checkpoint_store.CheckpointError, match="true or false"):
        checkpoint_store.pipeline_checkpoints_enabled({
            "PIPELINE_CHECKPOINTS_ENABLED": "yes",
        })


def test_canonical_json_normalizes_unicode_and_rejects_nonfinite():
    decomposed = {"value": "cancio\u0301n", "b": 1}
    composed = {"b": 1, "value": "canción"}
    assert checkpoint_store.canonical_json_bytes(decomposed) == \
        checkpoint_store.canonical_json_bytes(composed)
    with pytest.raises(checkpoint_store.CheckpointError):
        checkpoint_store.canonical_json_bytes({"bad": math.nan})


def test_disabled_store_reads_and_writes_nothing(tmp_path):
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(enabled=False), False, 1, _source(),
    )
    key = checkpoint_store.expected_asr_key(
        store.job_config, store.source_identity, "whisper-cpu",
    )
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    ) is False
    assert store.load_fixed("asr", key) is None
    assert not (tmp_path / ".checkpoints").exists()


def test_source_then_fixed_checkpoint_round_trip(tmp_path, monkeypatch):
    config = _config()
    store = checkpoint_store.CheckpointStore(str(tmp_path), config, True, 2)
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    identity = _source()
    request_key = checkpoint_store.expected_source_request_key(config)
    source_payload = {
        "source_fingerprint": identity,
        "artifact": {
            "relative_path": "source.mp4",
            "size": len(b"source"),
            "sha256": identity["sha256"],
            "duration_ms": identity["duration_ms"],
            "probe": {"duration": 1.0},
        },
        "retained_relative_path": "source.mp4",
        "probe": {"duration": 1.0, "width": 1920, "height": 1080},
    }
    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(
        checkpoint_store.subprocess, "run", lambda *args, **kwargs: Probe(),
    )
    assert store.save_source(request_key, source_payload, {})
    assert store.load_source(request_key)["payload"] == source_payload
    store.bind_source(identity)
    key = checkpoint_store.expected_asr_key(config, identity, "whisper-cpu")
    provenance = _asr_provenance("whisper-cpu")
    assert store.save_fixed(
        "asr", key, {"outcome": "speech", "transcript": _speech()},
        provenance,
    )
    loaded = store.load_fixed("asr", key)
    assert loaded["payload"]["outcome"] == "speech"
    assert loaded["writer_generation"] == 2


def test_completed_unpinned_whisper_stage_is_reusable(tmp_path):
    config = _config()
    config["asr"]["whisper"]["model_revision"] = None
    identity = _source()
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    key = checkpoint_store.expected_asr_key(
        config, identity, "whisper-cpu",
    )
    provenance = {
        "route_context": "whisper-cpu",
        "backend": {
            "effective_route": "whisper-cpu",
            "effective_backend": "whisper",
            "immutable_revision_available": False,
            "immutable_model_revision": None,
        },
    }
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "speech", "transcript": _speech()},
        provenance,
    )
    loaded = store.load_routed_fixed(
        "asr",
        lambda route: route,
        lambda route: checkpoint_store.expected_asr_key(
            config, identity, route,
        ),
    )
    assert loaded["payload"]["transcript"] == _speech()


def test_precomputed_transcript_stage_is_bound_to_accepted_hash(tmp_path):
    transcript_hash = checkpoint_store.sha256_bytes(b"transcript")
    config = _config()
    config["source"]["transcript_sha256"] = transcript_hash
    identity = _source()
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    key = checkpoint_store.expected_asr_key(
        config, identity, "precomputed-transcript",
    )
    provenance = {
        "route_context": "precomputed-transcript",
        "backend": {
            "effective_route": "precomputed-transcript",
            "effective_backend": "external",
            "immutable_revision_available": True,
            "immutable_model_revision": transcript_hash,
        },
    }
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "speech", "transcript": _speech()},
        provenance,
    )
    assert store.load_routed_fixed(
        "asr",
        lambda route: route,
        lambda route: checkpoint_store.expected_asr_key(
            config, identity, route,
        ),
    ) is not None


def test_same_name_and_duration_different_bytes_are_rejected(tmp_path):
    config = _config()
    original = _source(b"aaaa")
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, original,
    )
    key = checkpoint_store.expected_asr_key(config, original, "no-audio")
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    other = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 2, _source(b"bbbb"),
    )
    assert other.load_fixed("asr", key) is None


def test_small_probe_drift_is_accepted(tmp_path):
    config = _config()
    original = _source(duration_ms=1000)
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, original,
    )
    key = checkpoint_store.expected_asr_key(config, original, "no-audio")
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    drifted = dict(original, duration_ms=1300)
    resumed = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 2, drifted,
    )
    assert resumed.load_fixed("asr", key) is not None


def test_corrupt_payload_and_wrong_implementation_are_cache_misses(tmp_path):
    config = _config()
    identity = _source()
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    key = checkpoint_store.expected_asr_key(config, identity, "no-audio")
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    path = tmp_path / ".checkpoints" / "v1" / "asr.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["outcome"] = "speech"
    path.write_text(json.dumps(envelope))
    assert store.load_fixed("asr", key) is None
    envelope["payload_hash"] = checkpoint_store.sha256_bytes(
        checkpoint_store.canonical_json_bytes(envelope["payload"])
    )
    envelope["implementation_version"] = "asr-v999"
    path.write_text(json.dumps(envelope))
    assert store.load_fixed("asr", key) is None


def test_oversized_checkpoint_is_a_cache_miss(tmp_path, monkeypatch):
    config = _config()
    identity = _source()
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    key = checkpoint_store.expected_asr_key(config, identity, "no-audio")
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    monkeypatch.setattr(checkpoint_store, "MAX_CHECKPOINT_FILE_BYTES", 16)

    assert store.load_fixed("asr", key) is None


def test_routed_load_validates_route_before_calculating_expected_key(tmp_path):
    config = _config()
    identity = _source()
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    key = checkpoint_store.expected_asr_key(config, identity, "no-audio")
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    calls = []

    def validate(route):
        if route not in {"no-audio"}:
            raise ValueError("unknown")
        return route

    def expected(route):
        calls.append(route)
        return checkpoint_store.expected_asr_key(config, identity, route)

    assert store.load_routed_fixed("asr", validate, expected) is not None
    assert calls == ["no-audio"]
    path = tmp_path / ".checkpoints" / "v1" / "asr.json"
    envelope = json.loads(path.read_text())
    envelope["route_context"] = "../../bad"
    path.write_text(json.dumps(envelope))
    calls.clear()
    assert store.load_routed_fixed("asr", validate, expected) is None
    assert calls == []


def test_keyed_paths_use_only_expected_sha(tmp_path):
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(), True, 1, _source(),
    )
    with pytest.raises(checkpoint_store.CheckpointError):
        store.load_keyed("render", "../escape")


def test_checkpoint_root_symlink_cannot_redirect_writes(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".checkpoints").symlink_to(
        outside, target_is_directory=True,
    )
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(), True, 1, _source(),
    )
    key = checkpoint_store.expected_asr_key(
        store.job_config, store.source_identity, "no-audio",
    )

    with pytest.raises(
        checkpoint_store.CheckpointError,
        match="must not be a symlink",
    ):
        store.save_fixed(
            "asr",
            key,
            {"outcome": "no_audio", "transcript": None},
            _asr_provenance(),
        )
    assert list(outside.iterdir()) == []


def test_checkpoint_file_symlink_is_not_loaded(tmp_path):
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(), True, 1, _source(),
    )
    key = checkpoint_store.expected_asr_key(
        store.job_config, store.source_identity, "no-audio",
    )
    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "no_audio", "transcript": None},
        _asr_provenance(),
    )
    checkpoint_path = tmp_path / ".checkpoints" / "v1" / "asr.json"
    target = tmp_path / "outside.json"
    checkpoint_path.replace(target)
    checkpoint_path.symlink_to(target)

    assert store.load_fixed("asr", key) is None
