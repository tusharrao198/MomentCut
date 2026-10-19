import base64
import json
import multiprocessing
import os

import pytest

import job_state


def _config(job_id="job-1"):
    return job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "created_at": "2026-10-18T00:00:00Z",
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "checkpointing_enabled": True,
    })


def _state(job_id="job-1"):
    return {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "execution_generation": 2,
        "status": "processing",
    }


def _artifact(index=0):
    return {
        "clip_index": index,
        "relative_path": f"clip-{index}.mp4",
        "size": 10,
        "sha256": "a" * 64,
        "duration_ms": 1000,
        "probe": {"playable": True},
        "clip_plan_identity": "b" * 64,
        "companions": [],
    }


def test_atomic_job_files_round_trip_and_config_is_immutable(tmp_path):
    job_state.write_job_config(str(tmp_path), _config())
    job_state.write_job_state(str(tmp_path), _state())
    assert job_state.load_job_config(str(tmp_path))["job_id"] == "job-1"
    assert job_state.load_job_state(str(tmp_path))["execution_generation"] == 2
    changed = _config()
    changed["pipeline"] = {"captions": True}
    changed = job_state.seal_job_config(changed)
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_job_config(str(tmp_path), changed)
    assert not list(tmp_path.glob(".*.tmp"))


def test_required_load_distinguishes_missing_and_corrupt(tmp_path):
    with pytest.raises(job_state.StateFileMissing):
        job_state.load_job_state(str(tmp_path))
    (tmp_path / ".job-state.json").write_text("{", encoding="utf-8")
    with pytest.raises(job_state.StateFileInvalid):
        job_state.load_job_state(str(tmp_path))


def test_resume_manifest_rejects_plaintext_command_and_secret(tmp_path):
    safe = {
        "schema_version": 1,
        "job_id": "job-1",
        "configuration_revision": 1,
        "attempts": 0,
    }
    job_state.write_resume_manifest(str(tmp_path), safe)
    assert job_state.load_resume_manifest(str(tmp_path)) == safe
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_resume_manifest(
            str(tmp_path), {**safe, "cmd": ["python", "main.py"]}
        )
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_resume_manifest(
            str(tmp_path), {**safe, "webhook_secret": "secret"}
        )
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_resume_manifest(
            str(tmp_path), {**safe, "heartbeat": "not-a-number"}
        )
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_resume_manifest(
            str(tmp_path), {**safe, "priority": "high"}
        )


@pytest.mark.parametrize("path", [
    "../outside.mp4",
    r"..\outside.mp4",
    r"C:\outside.mp4",
    "/outside.mp4",
])
def test_job_config_rejects_cross_platform_unsafe_relative_paths(
        tmp_path, path):
    config = _config()
    config["source"]["retained_relative_path"] = path
    config = job_state.seal_job_config(config)
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_job_config(str(tmp_path), config)



def test_job_config_allows_boolean_llm_authentication_requirement(tmp_path):
    config = _config()
    config["credential_requirements"] = {"llm_authentication": True}
    config = job_state.seal_job_config(config)

    job_state.write_job_config(str(tmp_path), config)

    assert job_state.load_job_config(str(tmp_path)) == config

def test_resume_secret_keyring_validation():
    encoded = base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("=")
    ring = job_state.load_resume_secret_keyring({
        "RESUME_SECRET_ACTIVE_KEY_ID": "v1",
        "RESUME_SECRET_KEYS": f"v1:{encoded}",
    })
    assert ring["active_key_id"] == "v1"
    assert ring["keys"]["v1"] == b"k" * 32
    with pytest.raises(job_state.JobStateError):
        job_state.load_resume_secret_keyring({
            "RESUME_SECRET_ACTIVE_KEY_ID": "v2",
            "RESUME_SECRET_KEYS": f"v1:{encoded}",
        })



def test_resume_manifest_allows_boolean_api_caller(tmp_path):
    manifest = {
        "schema_version": 1,
        "job_id": "job-1",
        "configuration_revision": 1,
        "api_caller": True,
    }

    job_state.write_resume_manifest(str(tmp_path), manifest)


    assert job_state.load_resume_manifest(str(tmp_path)) == manifest
def test_resume_secrets_round_trip_and_authenticate_aad(tmp_path):
    pytest.importorskip("cryptography")
    encoded = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    ring = job_state.load_resume_secret_keyring({
        "RESUME_SECRET_ACTIVE_KEY_ID": "v1",
        "RESUME_SECRET_KEYS": f"v1:{encoded}",
    })
    job_state.write_resume_secrets(
        str(tmp_path), "job-1", 1, "webhook", {"secret": "value"}, ring,
    )
    assert job_state.load_resume_secrets(
        str(tmp_path), "job-1", 1, "webhook", ring,
    ) == {"secret": "value"}
    wrong_encoded = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    wrong_ring = job_state.load_resume_secret_keyring({
        "RESUME_SECRET_ACTIVE_KEY_ID": "v1",
        "RESUME_SECRET_KEYS": f"v1:{wrong_encoded}",
    })
    with pytest.raises(job_state.StateFileInvalid):
        job_state.load_resume_secrets(
            str(tmp_path), "job-1", 1, "webhook", wrong_ring,
        )
    with pytest.raises(job_state.StateFileInvalid):
        job_state.load_resume_secrets(
            str(tmp_path), "job-2", 1, "webhook", ring,
        )
    envelope = json.loads((tmp_path / ".resume-secrets.enc").read_text())
    ciphertext = envelope["ciphertext"]
    envelope["ciphertext"] = (
        "A" if ciphertext[0] != "A" else "B"
    ) + ciphertext[1:]
    (tmp_path / ".resume-secrets.enc").write_text(json.dumps(envelope))
    with pytest.raises(job_state.StateFileInvalid):
        job_state.load_resume_secrets(
            str(tmp_path), "job-1", 1, "webhook", ring,
        )


def test_job_lease_is_exclusive(tmp_path):
    first = job_state.JobLease.acquire(str(tmp_path), "job-1")
    try:
        with pytest.raises(job_state.JobLeaseBusy):
            job_state.JobLease.acquire(str(tmp_path), "job-1")
    finally:
        first.release()
    second = job_state.JobLease.acquire(str(tmp_path), "job-1")
    second.release()


def test_terminal_commit_is_generation_bound_and_allocates_complete_outbox(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())
    with pytest.raises(job_state.JobStateError):
        job_state.commit_terminal_state(
            str(tmp_path), 1, "completed", {}, [], {},
        )
    committed = job_state.commit_terminal_state(
        str(tmp_path),
        2,
        "partial",
        {"0": _artifact()},
        [1],
        {"actions": {"reservation": True, "webhook": True}},
    )
    assert committed["terminal_revision"] == 1
    assert committed["terminal_actions"]["reservation"]["state"] == "pending"
    assert committed["terminal_actions"]["webhook"]["state"] == "pending"
    assert committed["terminal_actions"]["archive"]["state"] == "not_required"
    assert len(committed["terminal_actions"]["webhook"]["event_id"]) == 64


def test_terminal_action_compare_and_set(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())
    job_state.commit_terminal_state(
        str(tmp_path), 2, "failed", {}, [], {"actions": {"reservation": True}},
    )
    state = job_state.transition_terminal_action(
        str(tmp_path), "reservation", "pending", "running",
    )
    assert state["terminal_actions"]["reservation"]["attempts"] == 1
    with pytest.raises(job_state.JobStateError):
        job_state.transition_terminal_action(
            str(tmp_path), "reservation", "pending", "done",
        )


def test_terminal_state_rejects_mutated_event_identity_and_context(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())
    committed = job_state.commit_terminal_state(
        str(tmp_path),
        2,
        "failed",
        {},
        [],
        {"actions": {"webhook": True}},
    )
    changed = json.loads(json.dumps(committed))
    changed["terminal_actions"]["webhook"]["event_id"] = "a" * 64
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_job_state(str(tmp_path), changed)

    changed = json.loads(json.dumps(committed))
    changed["terminal_action_context"]["status"] = "completed"
    with pytest.raises(job_state.StateFileInvalid):
        job_state.write_job_state(str(tmp_path), changed)


def test_read_json_rejects_oversized_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"value":"' + ("x" * 100) + '"}')

    assert job_state.read_json(str(path), max_bytes=32) is None
    assert job_state.read_json(str(path), max_bytes=1024) == {
        "value": "x" * 100,
    }

def test_terminal_state_allows_boolean_api_caller(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())

    committed = job_state.commit_terminal_state(
        str(tmp_path),
        2,
        "failed",
        {},
        [],
        {"actions": {}, "api_caller": True},
    )

    assert committed["terminal_action_context"]["api_caller"] is True
