"""The legacy basename/duration transcript cache is replaced by ASR envelopes."""
import json

import checkpoint_store


TRANSCRIPT = {
    "text": "hola",
    "language": "es",
    "segments": [
        {"start": 0.0, "end": 1.0, "text": "hola", "words": []},
    ],
}


def _config():
    return {
        "schema_version": 1,
        "job_id": "job-1",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "asr": {
            "requested_backend": "whisper",
            "whisper": {"model_revision": "whisper-rev-1"},
        },
        "checkpointing_enabled": True,
    }


def _identity(data):
    return {
        "sha256": checkpoint_store.sha256_bytes(data),
        "size": len(data),
        "duration_ms": 58640,
        "fps": 30.0,
    }


def test_speech_no_audio_and_sparse_speech_are_completed_outcomes(tmp_path):
    config = _config()
    source = _identity(b"video")
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, source,
    )
    for outcome, transcript, route in (
        ("speech", TRANSCRIPT, "whisper-cpu"),
        ("no_audio", None, "no-audio"),
        (
            "sparse_speech",
            {"text": "", "language": "en", "segments": []},
            "whisper-cpu",
        ),
    ):
        key = checkpoint_store.expected_asr_key(
            config, source, route,
        )
        backend = {
            "effective_route": route,
            "effective_backend": "none" if route == "no-audio" else "whisper",
        }
        if route != "no-audio":
            backend.update({
                "immutable_revision_available": True,
                "immutable_model_revision": "whisper-rev-1",
            })
        assert store.save_fixed(
            "asr",
            key,
            {"outcome": outcome, "transcript": transcript},
            {"route_context": route, "backend": backend},
        )
        assert store.load_fixed("asr", key)["payload"]["outcome"] == outcome

    def test_different_whisper_engine_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WHISPER_ENGINE", "faster-whisper")
        _save(tmp_path)
        monkeypatch.setenv("WHISPER_ENGINE", "auto")
        assert _load(tmp_path) is None

    def test_different_parakeet_engine_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
        monkeypatch.setenv("PARAKEET_ENGINE", "onnx")
        _save(tmp_path)
        monkeypatch.setenv("PARAKEET_ENGINE", "auto")
        assert _load(tmp_path) is None


def test_same_name_and_duration_cannot_substitute_different_bytes(tmp_path):
    config = _config()
    source_a = _identity(b"AAAA")
    source_b = _identity(b"BBBB")
    first = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, source_a,
    )
    key = checkpoint_store.expected_asr_key(
        config, source_a, "whisper-cpu",
    )
    first.save_fixed(
        "asr",
        key,
        {"outcome": "speech", "transcript": TRANSCRIPT},
        {
            "route_context": "whisper-cpu",
            "backend": {
                "effective_route": "whisper-cpu",
                "effective_backend": "whisper",
                "immutable_revision_available": True,
                "immutable_model_revision": "whisper-rev-1",
            },
        },
    )
    resumed = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 2, source_b,
    )
    assert resumed.load_fixed("asr", key) is None


def test_small_adjacent_segment_overlap_is_a_complete_asr_stage(tmp_path):
    config = _config()
    config["asr"]["whisper"]["model_revision"] = None
    source = _identity(b"overlap-video")
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, source,
    )
    transcript = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "first", "words": []},
            {"start": 9.8, "end": 20.0, "text": "second", "words": []},
        ],
    }
    key = checkpoint_store.expected_asr_key(
        config, source, "whisper-cpu",
    )

    assert store.save_fixed(
        "asr",
        key,
        {"outcome": "speech", "transcript": transcript},
        {
            "route_context": "whisper-cpu",
            "backend": {
                "effective_route": "whisper-cpu",
                "effective_backend": "whisper",
                "immutable_revision_available": False,
                "immutable_model_revision": None,
            },
        },
    )


def test_materially_backwards_asr_segments_are_not_checkpointed(tmp_path):
    config = _config()
    config["asr"]["whisper"]["model_revision"] = None
    source = _identity(b"backwards-video")
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, source,
    )
    transcript = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "first", "words": []},
            {"start": 8.0, "end": 20.0, "text": "second", "words": []},
        ],
    }
    key = checkpoint_store.expected_asr_key(
        config, source, "whisper-cpu",
    )

    assert not store.save_fixed(
        "asr",
        key,
        {"outcome": "speech", "transcript": transcript},
        {
            "route_context": "whisper-cpu",
            "backend": {
                "effective_route": "whisper-cpu",
                "effective_backend": "whisper",
                "immutable_revision_available": False,
                "immutable_model_revision": None,
            },
        },
    )


def test_legacy_transcript_checkpoint_is_ignored(tmp_path):
    legacy = tmp_path / ".transcript_checkpoint.json"
    legacy.write_text(json.dumps({
        "source": {"name": "video.mp4", "duration": 58.6},
        "transcript": TRANSCRIPT,
    }))
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(), True, 1, _identity(b"video"),
    )
    key = checkpoint_store.expected_asr_key(
        store.job_config, store.source_identity, "whisper-cpu",
    )
    assert store.load_fixed("asr", key) is None
