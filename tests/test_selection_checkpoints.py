import checkpoint_store


def _project(tmp_path):
    config = {
        "schema_version": 1,
        "job_id": "selection-job",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "asr": {"requested_backend": "whisper"},
        "selection": {},
        "checkpointing_enabled": True,
    }
    identity = {
        "sha256": checkpoint_store.sha256_bytes(b"source"),
        "size": 6,
        "duration_ms": 60_000,
        "fps": 30.0,
    }
    return config, identity, checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )


def test_direct_clip_plan_load_does_not_require_asr_file(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    assert store.save_fixed(
        "clip-plan",
        key,
        {
            "shorts": [{"start": 0.0, "end": 15.0}],
            "transcript": {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 15.0,
                        "text": "hello",
                        "words": [],
                    },
                ],
            },
            "source_analysis": {
                "source_subtitle_band": None,
                "layout_config": [],
                "checkpoints": {},
            },
        },
        {"route_context": route},
    )
    assert not (
        tmp_path / ".checkpoints" / "v1" / "asr.json"
    ).exists()
    loaded = store.load_current_clip_plan(
        lambda saved_route: checkpoint_store.expected_clip_plan_key(
            config, identity, saved_route,
        )
    )
    assert loaded["payload"]["shorts"][0]["end"] == 15.0


def test_completed_cloud_clip_plan_is_reusable_without_model_revision(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-gemini",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {
        "shorts": [{"start": 0.0, "end": 15.0}],
        "transcript": {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "hello", "words": []},
            ],
        },
        "source_analysis": {
            "source_subtitle_band": None,
            "layout_config": [],
            "checkpoints": {},
        },
    }
    assert store.save_fixed(
        "clip-plan", key, payload, {"route_context": route},
    )
    loaded = store.load_current_clip_plan(
        lambda saved_route: checkpoint_store.expected_clip_plan_key(
            config, identity, saved_route,
        )
    )
    assert loaded["payload"] == payload


def test_clip_plan_rejects_invalid_embedded_source_analysis(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {
        "shorts": [{"start": 0.0, "end": 15.0}],
        "transcript": {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "hello", "words": []},
            ],
        },
        # Presence means the plan claims this render decision is authoritative;
        # null/malformed data must therefore invalidate the plan.
        "source_analysis": None,
    }

    assert not store.save_fixed(
        "clip-plan", key, payload, {"route_context": route},
    )


def test_clip_plan_rejects_missing_source_analysis_snapshot(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)

    assert not store.save_fixed(
        "clip-plan",
        key,
        {
            "shorts": [{"start": 0.0, "end": 15.0}],
            "transcript": {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 15.0,
                        "text": "hello",
                        "words": [],
                    },
                ],
            },
        },
        {"route_context": route},
    )


def test_clip_plan_keeps_failed_best_effort_analysis_for_exact_render_replay(
        tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {
        "shorts": [{"start": 0.0, "end": 15.0}],
        "transcript": {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "hello", "words": []},
            ],
        },
        "source_analysis": {
            "source_subtitle_band": None,
            "layout_config": [],
            "checkpoints": {
                "layout": {
                    "compatibility_key": "a" * 64,
                    "result": {
                        "outcome": "failure",
                        "reason": "timeout",
                        "layout": "none",
                        "applied_flags": [
                            "split_layout", "active_speaker",
                        ],
                    },
                },
            },
        },
    }

    assert store.save_fixed(
        "clip-plan", key, payload, {"route_context": route},
    )
    loaded = store.load_current_clip_plan(
        lambda saved_route: checkpoint_store.expected_clip_plan_key(
            config, identity, saved_route,
        )
    )
    assert loaded["payload"]["source_analysis"] == payload["source_analysis"]


def test_clip_plan_rejects_mismatched_subtitle_snapshot(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    payload = {
        "shorts": [{"start": 0.0, "end": 15.0}],
        "transcript": {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "hello", "words": []},
            ],
        },
        "source_analysis": {
            "source_subtitle_band": 0.8,
            "layout_config": [],
            "checkpoints": {
                "subtitles": {
                    "compatibility_key": "a" * 64,
                    "result": {"outcome": "none", "band": None},
                },
            },
        },
    }

    assert not store.save_fixed(
        "clip-plan", key, payload, {"route_context": route},
    )


def test_clip_plan_outside_configured_duration_is_rejected(tmp_path):
    config, identity, store = _project(tmp_path)
    config["selection"] = {
        "clip_min_seconds": "15",
        "clip_max_seconds": "60",
    }
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": "sha256:model-v1",
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    assert not store.save_fixed(
        "clip-plan",
        key,
        {
            "shorts": [{"start": 0.0, "end": 5.0}],
            "transcript": {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "hello", "words": []},
                ],
            },
        },
        {"route_context": route},
    )


def test_unknown_clip_plan_route_is_rejected_before_key_builder(tmp_path):
    config, identity, store = _project(tmp_path)
    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": "sha256:model-v1",
    }
    key = checkpoint_store.expected_clip_plan_key(config, identity, route)
    store.save_fixed(
        "clip-plan", key, {"shorts": []},
        {"route_context": {**route, "selection_route": "../../bad"}},
    )
    called = []
    assert store.load_current_clip_plan(
        lambda saved_route: called.append(saved_route) or key
    ) is None
    assert called == []
