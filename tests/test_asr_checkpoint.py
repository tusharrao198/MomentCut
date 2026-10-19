import transcribe_backends as tb
from types import SimpleNamespace


def _transcript():
    return {
        "text": "hello",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"word": " hello", "start": 0.0, "end": 0.5}],
            },
        ],
    }


def test_no_audio_is_a_structured_completed_outcome(monkeypatch):
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _path: False)
    result = tb.transcribe_media_with_provenance("silent.mp4")
    assert result["outcome"] == "no_audio"
    assert result["transcript"] is None
    assert result["provenance"]["effective_route"] == "no-audio"


def test_parakeet_rejection_records_effective_whisper_route(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setenv("PARAKEET_ENGINE", "onnx")
    rejected = _transcript()
    rejected["language"] = "ja"
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _path: True)
    monkeypatch.setattr(tb, "_transcribe_with_selected_parakeet", lambda _path: rejected)
    monkeypatch.setattr(
        tb, "_transcribe_with_whisper_execution",
        lambda _path: (_transcript(), {"engine": "faster-whisper", "force_cpu": False}),
    )
    monkeypatch.setattr(tb, "_whisper_device", lambda: "cpu")
    result = tb.transcribe_media_with_provenance("video.mp4")
    provenance = result["provenance"]
    assert provenance["effective_route"] == "parakeet-to-whisper-cpu"
    assert provenance["fallback_reason_code"] == "parakeet_unsupported_language"
    assert provenance["attempts"][0] == {
        "backend": "parakeet", "engine": "onnx", "outcome": "rejected",
    }


def test_parakeet_does_not_claim_unenforced_revision(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setenv("PARAKEET_MODEL_REVISION", "a" * 40)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _path: True)
    monkeypatch.setenv("PARAKEET_ENGINE", "onnx")
    monkeypatch.setattr(
        tb, "_transcribe_with_selected_parakeet", lambda _path: _transcript()
    )

    result = tb.transcribe_media_with_provenance("video.mp4")

    provenance = result["provenance"]
    assert provenance["effective_route"] == "parakeet"
    assert provenance["immutable_revision_available"] is False
    assert provenance["immutable_model_revision"] is None


def test_mlx_whisper_provenance_records_the_pinned_apple_route(monkeypatch):
    import accelerator_runtime

    revision = "a" * 40
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisper")
    monkeypatch.setenv("WHISPER_ENGINE", "auto")
    monkeypatch.setattr(
        accelerator_runtime, "is_apple_silicon", lambda **_kwargs: True)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _path: True)
    monkeypatch.setattr(
        tb,
        "_transcribe_with_whisper_execution",
        lambda _path: (_transcript(), {"engine": "mlx", "force_cpu": False}),
    )
    monkeypatch.setattr(
        tb,
        "_mac_asr",
        lambda: SimpleNamespace(
            resolve_whisper_model=lambda: ("mlx-community/whisper-small-mlx", revision)
        ),
    )

    provenance = tb.transcribe_media_with_provenance("video.mp4")["provenance"]

    assert provenance["effective_route"] == "whisper-mlx"
    assert provenance["immutable_model_revision"] == revision
    assert provenance["device"] == "mlx"


def test_mlx_parakeet_provenance_records_the_pinned_apple_route(monkeypatch):
    import accelerator_runtime

    revision = "b" * 40
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setenv("PARAKEET_ENGINE", "auto")
    monkeypatch.setattr(
        accelerator_runtime, "is_apple_silicon", lambda **_kwargs: True)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _path: True)
    monkeypatch.setattr(
        tb, "_transcribe_with_selected_parakeet", lambda _path: _transcript())
    monkeypatch.setattr(
        tb,
        "_mac_asr",
        lambda: SimpleNamespace(
            resolve_parakeet_model=lambda: ("mlx-community/parakeet-tdt-0.6b-v3", revision)
        ),
    )

    provenance = tb.transcribe_media_with_provenance("video.mp4")["provenance"]

    assert provenance["effective_route"] == "parakeet-mlx"
    assert provenance["immutable_model_revision"] == revision
    assert provenance["device"] == "mlx"


def test_checkpoint_asr_config_uses_mlx_model_revisions(monkeypatch):
    import accelerator_runtime

    whisper_revision = "c" * 40
    parakeet_revision = "d" * 40
    monkeypatch.setenv("WHISPER_ENGINE", "auto")
    monkeypatch.setenv("PARAKEET_ENGINE", "auto")
    monkeypatch.setattr(
        accelerator_runtime, "is_apple_silicon", lambda **_kwargs: True)
    monkeypatch.setattr(
        tb,
        "_mac_asr",
        lambda: SimpleNamespace(
            resolve_whisper_model=lambda: ("mlx-community/whisper-small-mlx", whisper_revision),
            resolve_parakeet_model=lambda: ("mlx-community/parakeet-tdt-0.6b-v3", parakeet_revision),
        ),
    )

    config = tb.checkpoint_asr_config()

    assert config["whisper"]["model_revision"] == whisper_revision
    assert config["parakeet"]["model_revision"] == parakeet_revision


def test_coordinator_uses_the_resolved_asr_checkpoint_config(monkeypatch, tmp_path):
    import app

    resolved = {
        "requested_backend": "whisper",
        "fallback_policy": "parakeet-to-whisper-route-v1",
        "whisper": {
            "engine": "mlx",
            "model": "mlx-community/whisper-small-mlx",
            "model_revision": "a" * 40,
            "device": "mlx",
            "compute": "mlx",
        },
        "parakeet": {
            "engine": "mlx",
            "model": "mlx-community/parakeet-tdt-0.6b-v3",
            "model_revision": "b" * 40,
            "vad_batch": "4",
        },
        "sparse_policy": {"min_words": "8", "min_words_per_minute": "5"},
    }
    monkeypatch.setattr(tb, "checkpoint_asr_config", lambda: resolved)
    monkeypatch.setattr(app.llm_backend, "active", lambda: False)
    monkeypatch.setattr(app.vision_backend, "active", lambda: False)

    config = app._resolved_job_config(
        "job", str(tmp_path), {"kind": "upload", "retained_relative_path": "source.mp4"},
        {}, "auto", [], auto_hook=False, auto_hook_style=None, captions=True,
        target_clips=None, clip_min_seconds=None, clip_max_seconds=None,
        checkpointing_enabled=True,
    )

    assert config["asr"] == resolved


def test_checkpoint_environment_carries_all_visual_decode_controls():
    import checkpoint_store

    assert {
        "VISION_CANDIDATES_PER_WINDOW",
        "VISION_SCENE_SNAP_SECONDS",
        "GPU_DECODE_MIN_FREE_MB",
    } <= set(checkpoint_store.PIPELINE_ENV_NAMES)
