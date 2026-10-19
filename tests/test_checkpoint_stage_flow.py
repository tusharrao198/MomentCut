import checkpoint_store


def _transcript():
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 60.0,
                "text": "A complete timestamped transcript.",
                "words": [],
            },
        ],
    }


def _config():
    return {
        "schema_version": 1,
        "job_id": "flow-job",
        "configuration_revision": 1,
        "source": {
            "kind": "upload",
            "retained_relative_path": "source.mp4",
        },
        "pipeline": {"output_format": "vertical"},
        "asr": {
            "requested_backend": "whisper",
            "whisper": {"model": "small", "model_revision": None},
        },
        "selection": {
            "clip_min_seconds": 15,
            "clip_max_seconds": 60,
            "target_max": 8,
        },
        "render": {"encoder": "libx264"},
        "checkpointing_enabled": True,
    }


def _probe_ok(*_args, **_kwargs):
    class Probe:
        returncode = 0
        stdout = "120.0\n"

    return Probe()


def test_real_life_restart_starts_at_first_incomplete_stage(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(checkpoint_store.subprocess, "run", _probe_ok)
    config = _config()
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-video")
    source_identity = {
        "sha256": checkpoint_store.sha256_file(str(source_path)),
        "size": source_path.stat().st_size,
        "duration_ms": 120_000,
        "fps": 30.0,
    }
    source_payload = {
        "source_fingerprint": source_identity,
        "retained_relative_path": "source.mp4",
        "probe": {"duration": 120.0, "width": 1920, "height": 1080},
        "artifact": {
            "relative_path": "source.mp4",
            "size": source_identity["size"],
            "sha256": source_identity["sha256"],
            "duration_ms": source_identity["duration_ms"],
            "probe": {"duration": 120.0},
        },
    }

    first = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1,
    )
    source_key = checkpoint_store.expected_source_request_key(config)
    assert first.save_source(source_key, source_payload, {})
    first.bind_source(source_identity)

    # A crashed/partial ASR result is rejected and therefore cannot make the
    # next run skip Whisper.
    asr_key = checkpoint_store.expected_asr_key(
        config, source_identity, "whisper-cpu",
    )
    assert not first.save_fixed(
        "asr",
        asr_key,
        {"outcome": "speech", "transcript": {"language": "en"}},
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

    second = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 2,
    )
    assert second.load_source(source_key) is not None
    second.bind_source(source_identity)
    assert second.load_routed_fixed(
        "asr",
        lambda route: route,
        lambda route: checkpoint_store.expected_asr_key(
            config, source_identity, route,
        ),
    ) is None

    # Once Whisper fully completes, its result is reusable even without a
    # pinned model revision because this is the same sealed job and source.
    asr_provenance = {
        "route_context": "whisper-cpu",
        "backend": {
            "effective_route": "whisper-cpu",
            "effective_backend": "whisper",
            "immutable_revision_available": False,
            "immutable_model_revision": None,
        },
    }
    assert second.save_fixed(
        "asr",
        asr_key,
        {"outcome": "speech", "transcript": _transcript()},
        asr_provenance,
    )

    route = {
        "selection_route": "speech-local",
        "asr_effective_route": "whisper-cpu",
        "asr_outcome": "speech",
        "selection_model_revision": None,
    }
    plan_key = checkpoint_store.expected_clip_plan_key(
        config, source_identity, route,
    )
    plan_payload = {
        "shorts": [
            {"start": 0.0, "end": 20.0},
            {"start": 40.0, "end": 60.0},
        ],
        "transcript": _transcript(),
        "source_analysis": {
            "source_subtitle_band": None,
            "layout_config": [],
            "checkpoints": {},
        },
    }
    assert second.save_fixed(
        "clip-plan", plan_key, plan_payload, {"route_context": route},
    )
    plan = second.load_current_clip_plan(
        lambda saved_route: checkpoint_store.expected_clip_plan_key(
            config, source_identity, saved_route,
        )
    )
    assert plan is not None

    first_clip = plan_payload["shorts"][0]
    render_key = checkpoint_store.expected_render_key(
        config, source_identity, plan, first_clip,
        plan["payload"]["source_analysis"],
    )
    rendered_path = tmp_path / "clip-1.mp4"
    rendered_path.write_bytes(b"complete-render")
    artifact = {
        "relative_path": "clip-1.mp4",
        "size": rendered_path.stat().st_size,
        "sha256": checkpoint_store.sha256_file(str(rendered_path)),
        "duration_ms": 20_000,
        "probe": {"duration": 20.0},
    }
    assert second.save_keyed(
        "render",
        render_key,
        {
            "clip_identity": checkpoint_store.sha256_bytes(
                checkpoint_store.canonical_json_bytes(first_clip)
            ),
            "display_index": 0,
            "artifact": artifact,
            "clip_metadata": {},
        },
        {"upstream_compatibility_keys": [plan_key]},
    )

    # A partial second output exists but has no completed render checkpoint.
    (tmp_path / "temp_clip-2.mp4").write_bytes(b"partial-render")

    third = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 3, source_identity,
    )
    resumed_plan = third.load_current_clip_plan(
        lambda saved_route: checkpoint_store.expected_clip_plan_key(
            config, source_identity, saved_route,
        )
    )
    assert resumed_plan is not None
    assert set(third.valid_render_artifacts(resumed_plan)) == {0}
