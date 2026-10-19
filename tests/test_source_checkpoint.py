import checkpoint_store


def _config():
    return {
        "schema_version": 1,
        "job_id": "source-job",
        "configuration_revision": 1,
        "source": {
            "kind": "upload",
            "retained_relative_path": "source.mp4",
            "original_name": "display-only.mp4",
        },
        "pipeline": {},
        "checkpointing_enabled": True,
    }


def test_source_checkpoint_is_bound_to_retained_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original source")
    identity = checkpoint_store.build_source_identity(
        str(source), {"duration": 2.0, "fps": 30.0},
    )
    config = _config()
    store = checkpoint_store.CheckpointStore(str(tmp_path), config, True, 1)
    request_key = checkpoint_store.expected_source_request_key(config)
    assert store.save_source(request_key, {
        "source_fingerprint": identity,
        "artifact": {
            "relative_path": "source.mp4",
            "size": identity["size"],
            "sha256": identity["sha256"],
            "duration_ms": identity["duration_ms"],
            "probe": {"duration": 2.0},
        },
        "retained_relative_path": "source.mp4",
        "probe": {"duration": 2.0, "width": 1920, "height": 1080},
    }, {})
    class Probe:
        returncode = 0
        stdout = "2.0\n"

    monkeypatch.setattr(
        checkpoint_store.subprocess, "run", lambda *args, **kwargs: Probe(),
    )
    assert store.load_source(request_key) is not None

    source.write_bytes(b"different bytes")
    assert store.load_source(request_key) is None


def test_display_name_does_not_change_source_request_identity():
    left = _config()
    right = _config()
    right["source"]["original_name"] = "renamed.mov"
    assert checkpoint_store.expected_source_request_key(left) == \
        checkpoint_store.expected_source_request_key(right)


def test_source_checkpoint_requires_analyzable_video_dimensions(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not-enough-to-prove-video")
    identity = checkpoint_store.build_source_identity(
        str(source), {"duration": 2.0, "fps": 30.0},
    )
    store = checkpoint_store.CheckpointStore(
        str(tmp_path), _config(), True, 1,
    )

    assert not store.save_source(
        checkpoint_store.expected_source_request_key(_config()),
        {
            "source_fingerprint": identity,
            "artifact": {
                "relative_path": "source.mp4",
                "size": identity["size"],
                "sha256": identity["sha256"],
                "duration_ms": identity["duration_ms"],
                "probe": {"duration": 2.0},
            },
            "retained_relative_path": "source.mp4",
            "probe": {"duration": 2.0, "width": 0, "height": 0},
        },
        {},
    )
