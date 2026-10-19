import checkpoint_store
import job_state


def test_open_existing_project_requires_matching_durable_state(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    config = job_state.seal_job_config({
        "schema_version": 1,
        "job_id": "recover-job",
        "project_dir": str(tmp_path),
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "checkpointing_enabled": True,
    })
    state = {
        "schema_version": 1,
        "job_id": "recover-job",
        "configuration_revision": 1,
        "execution_generation": 2,
        "status": "processing",
    }
    job_state.write_job_config(str(tmp_path), config)
    job_state.write_job_state(str(tmp_path), state)
    store = checkpoint_store.CheckpointStore(str(tmp_path), config, True, 2)
    identity = checkpoint_store.build_source_identity(
        str(source), {"duration": 1.0, "fps": 30.0},
    )
    store.save_source(
        checkpoint_store.expected_source_request_key(config),
        {
            "source_fingerprint": identity,
            "artifact": {
                "relative_path": "source.mp4",
                "size": identity["size"],
                "sha256": identity["sha256"],
                "duration_ms": identity["duration_ms"],
                "probe": {"duration": 1.0},
            },
            "retained_relative_path": "source.mp4",
            "probe": {"duration": 1.0, "width": 1920, "height": 1080},
        },
        {},
    )
    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(
        checkpoint_store.subprocess, "run", lambda *args, **kwargs: Probe(),
    )
    resumed = checkpoint_store.CheckpointStore.open_existing_project(
        str(tmp_path), current_enablement=True,
    )
    assert resumed.execution_generation == 2
    assert resumed.source_identity["sha256"] == identity["sha256"]
    assert checkpoint_store.completed_selection_boundary(
        config, current_enablement=True,
    ) is None

    route = "no-audio"
    assert resumed.save_fixed(
        "asr",
        checkpoint_store.expected_asr_key(config, identity, route),
        {"outcome": "no_audio", "transcript": None},
        {
            "route_context": route,
            "backend": {
                "effective_route": route,
                "effective_backend": "none",
            },
        },
    )
    assert checkpoint_store.completed_selection_boundary(
        config, current_enablement=True,
    ) == "silent"


def test_open_existing_project_does_not_retroactively_enable_checkpoints(
    tmp_path,
):
    config = job_state.seal_job_config({
        "schema_version": 1,
        "job_id": "disabled-job",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "checkpointing_enabled": False,
    })
    state = {
        "schema_version": 1,
        "job_id": "disabled-job",
        "configuration_revision": 1,
        "execution_generation": 1,
        "status": "processing",
    }
    job_state.write_job_config(str(tmp_path), config)
    job_state.write_job_state(str(tmp_path), state)

    resumed = checkpoint_store.CheckpointStore.open_existing_project(
        str(tmp_path), current_enablement=True,
    )
    assert resumed.enabled is False
    assert resumed.source_identity is None


def test_two_recovery_claimants_have_one_lease_owner(tmp_path):
    first = job_state.JobLease.acquire(str(tmp_path), "recover-job")
    try:
        try:
            job_state.JobLease.acquire(str(tmp_path), "recover-job")
        except job_state.JobLeaseBusy:
            pass
        else:
            raise AssertionError("second recovery claimant acquired the lease")
    finally:
        first.release()
