import checkpoint_store


def test_only_hash_and_probe_validated_render_is_reusable(tmp_path, monkeypatch):
    config = {
        "schema_version": 1,
        "job_id": "render-job",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {"output_format": "vertical"},
        "render": {"encoder": "x264"},
        "checkpointing_enabled": True,
    }
    identity = {
        "sha256": checkpoint_store.sha256_bytes(b"source"),
        "size": 6,
        "duration_ms": 1000,
        "fps": 30.0,
    }
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    plan = {
        "compatibility_key": "a" * 64,
        "payload_hash": "b" * 64,
        "payload": {
            "shorts": [{"start": 0.0, "end": 1.0}],
            "source_analysis": {},
        },
    }
    key = checkpoint_store.expected_render_key(
        config, identity, plan, plan["payload"]["shorts"][0], {},
    )
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"rendered")
    artifact = {
        "relative_path": "clip.mp4",
        "size": clip.stat().st_size,
        "sha256": checkpoint_store.sha256_file(str(clip)),
        "duration_ms": 1000,
        "probe": {"playable": True},
    }

    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(checkpoint_store.subprocess, "run", lambda *a, **k: Probe())
    assert store.save_keyed("render", key, {
        "clip_identity": checkpoint_store.sha256_bytes(
            checkpoint_store.canonical_json_bytes(
                plan["payload"]["shorts"][0]
            )
        ),
        "display_index": 0,
        "artifact": artifact,
        "clip_metadata": {},
    }, {})
    assert set(store.valid_render_artifacts(plan)) == {0}
    clip.write_bytes(b"tampered")
    assert store.valid_render_artifacts(plan) == {}


def test_truncated_playable_render_is_not_reusable(tmp_path, monkeypatch):
    config = {
        "schema_version": 1,
        "job_id": "truncated-render-job",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {"output_format": "vertical"},
        "render": {"encoder": "x264"},
        "checkpointing_enabled": True,
    }
    identity = {
        "sha256": checkpoint_store.sha256_bytes(b"source"),
        "size": 6,
        "duration_ms": 60_000,
        "fps": 30.0,
    }
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )
    planned_clip = {"start": 10.0, "end": 30.0}
    plan = {
        "compatibility_key": "a" * 64,
        "payload_hash": "b" * 64,
        "payload": {"shorts": [planned_clip], "source_analysis": {}},
    }
    key = checkpoint_store.expected_render_key(
        config, identity, plan, planned_clip, {},
    )
    clip = tmp_path / "truncated.mp4"
    clip.write_bytes(b"playable-but-short")

    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(
        checkpoint_store.subprocess, "run", lambda *args, **kwargs: Probe(),
    )
    assert store.save_keyed("render", key, {
        "clip_identity": checkpoint_store.sha256_bytes(
            checkpoint_store.canonical_json_bytes(planned_clip)
        ),
        "display_index": 0,
        "artifact": {
            "relative_path": clip.name,
            "size": clip.stat().st_size,
            "sha256": checkpoint_store.sha256_file(str(clip)),
            "duration_ms": 1_000,
            "probe": {"playable": True},
        },
        "clip_metadata": {},
    }, {})

    assert store.valid_render_artifacts(plan) == {}
