"""Transient failures retry by themselves; a shutdown never fails a job.

22-sep-2026: 30-60% of jobs per hour failed at peak on a full GPU, and a
deploy's drain timeout marked six running jobs failed and refunded at once.
Users saw "failed" for problems that were entirely ours.
"""
import asyncio
import json
import os

import pytest

app_module = pytest.importorskip("app")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_stopping", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "AUTO_RETRY_DELAY_SECONDS", 0)
    calls = {"settled": [], "enqueued": []}

    async def fake_settle(job_id, job=None):
        calls["settled"].append(job_id)
    for name in ("_archive_managed_job", "_notify_job_webhook", "_record_job_alert",
                 "_track_proxy_usage", "_notify_clips_ready", "_notify_clip_activity"):
        async def _noop(*a, **k):
            return None
        monkeypatch.setattr(app_module, name, _noop)

    async def _noop2(*a, **k):
        return None
    monkeypatch.setattr(app_module, "_autopilot_finalize_and_prepare", _noop2)
    monkeypatch.setattr(app_module, "_settle_reservation", fake_settle)
    monkeypatch.setattr(app_module, "_enqueue_job",
                        lambda job_id, priority=2: calls["enqueued"].append((job_id, priority)))
    return root, calls


def _job(root, job_id, logs):
    d = root / job_id
    d.mkdir()
    (d / app_module._RESUME_FILE).write_text(json.dumps({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "priority": 1,
        "attempts": 0,
    }))
    (d / "My_Video.mp4").write_bytes(b"src")
    (d / ".transcript_checkpoint.json").write_text("{}")
    (d / "My_Video_metadata.json").write_text("{}")
    (d / "My_Video_clip_1.mp4").write_bytes(b"half")
    job = {"status": "queued", "logs": list(logs), "cmd": ["x"], "env": {},
           "output_dir": str(d), "user_id": "u", "reservation_id": "r"}
    app_module.jobs[job_id] = job
    return d, job


def _run_wrapper(monkeypatch, job_id, final_status, raise_cancel=False):
    async def fake_run_job(jid, job):
        if raise_cancel:
            raise asyncio.CancelledError()
        job["status"] = final_status

    monkeypatch.setattr(app_module, "run_job", fake_run_job)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        monkeypatch.setattr(app_module, "concurrency_semaphore", sem)
        q = asyncio.PriorityQueue()
        q.put_nowait((1, 0, job_id))
        await q.get()
        monkeypatch.setattr(app_module, "job_queue", q)
        try:
            await app_module.run_job_wrapper(job_id)
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)   # let the scheduled retry fire
        return sem
    return asyncio.run(go())


class TestClassification:
    def test_gpu_oom_is_transient(self):
        assert app_module.is_transient_failure(
            ["❌ Clip 1 failed: RuntimeError: CUDA out of memory"])

    def test_nvenc_is_transient(self):
        assert app_module.is_transient_failure(
            ["ERROR: Generic error in an external library", "Process failed with exit code 187"])

    def test_content_problems_are_not(self):
        assert not app_module.is_transient_failure(
            ["❌ RuntimeError: Clip detection failed — the AI model did not return usable clips"])
        assert not app_module.is_transient_failure(["ERROR: [youtube] x: Video unavailable"])
        assert not app_module.is_transient_failure(["❌ This video has no audio track."])

    def test_unknown_is_not(self):
        assert not app_module.is_transient_failure(["Process failed with exit code 1"])


class TestWrapper:
    def test_transient_failure_is_retried_once(self, env, monkeypatch):
        root, calls = env
        d, job = _job(root, "j1", ["❌ CUDA out of memory"])
        sem = _run_wrapper(monkeypatch, "j1", "failed")
        assert job["status"] == "queued" and job["auto_retries"] == 1
        assert calls["enqueued"] == [("j1", 1)]
        assert calls["settled"] == []                         # reservation kept
        assert (d / app_module._RESUME_FILE).exists()         # still resumable
        assert (d / "My_Video.mp4").exists() and (d / ".transcript_checkpoint.json").exists()
        assert not (d / "My_Video_clip_1.mp4").exists()
        assert not (d / "My_Video_metadata.json").exists()
        assert sem._value == 1                                # slot released

    def test_second_failure_is_final(self, env, monkeypatch):
        root, calls = env
        d, job = _job(root, "j2", ["❌ CUDA out of memory"])
        job["auto_retries"] = 1
        _run_wrapper(monkeypatch, "j2", "failed")
        assert job["status"] == "failed"
        assert calls["settled"] == ["j2"] and calls["enqueued"] == []
        assert not (d / app_module._RESUME_FILE).exists()

    def test_content_failure_is_not_retried(self, env, monkeypatch):
        root, calls = env
        d, job = _job(root, "j3", ["❌ Clip detection failed"])
        _run_wrapper(monkeypatch, "j3", "failed")
        assert job["status"] == "failed" and calls["enqueued"] == []

    def test_shutdown_leaves_the_job_for_the_next_instance(self, env, monkeypatch):
        root, calls = env
        d, job = _job(root, "j4", [])
        sem = _run_wrapper(monkeypatch, "j4", None, raise_cancel=True)
        assert calls["settled"] == []                         # no refund, no failure
        assert (d / app_module._RESUME_FILE).exists()
        assert sem._value == 1

    def test_busy_lease_leaves_durable_job_nonterminal(self, env, monkeypatch):
        root, calls = env
        d, job = _job(root, "lease-busy", [])
        config = app_module.job_state.seal_job_config({
            "schema_version": 1,
            "job_id": "lease-busy",
            "configuration_revision": 1,
            "project_dir": str(d),
            "source": {
                "kind": "upload",
                "retained_relative_path": "My_Video.mp4",
            },
            "pipeline": {},
            "checkpointing_enabled": False,
        })
        app_module.job_state.write_job_config(str(d), config)
        app_module.job_state.write_job_state(str(d), {
            "schema_version": 1,
            "job_id": "lease-busy",
            "configuration_revision": 1,
            "execution_generation": 2,
            "status": "queued",
        })
        held = app_module.job_state.JobLease.acquire(str(d), "lease-busy")
        try:
            sem = _run_wrapper(monkeypatch, "lease-busy", "failed")
        finally:
            held.release()
        state = app_module.job_state.load_job_state(str(d))
        assert state["status"] == "queued"
        assert state["execution_generation"] == 2
        assert (d / app_module._RESUME_FILE).exists()
        assert calls["settled"] == []
        assert "lease-busy" not in app_module.jobs
        assert sem._value == 1


class TestStructuredWorkerFailures:
    def test_disabled_records_preserve_legacy_retry_policy(self, env, monkeypatch):
        root, _calls = env
        _directory, job = _job(root, "n9-legacy", ["❌ CUDA out of memory"])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: False)

        assert app_module._prepare_worker_failure("n9-legacy", job) is None
        job["status"] = "failed"
        assert app_module._should_auto_retry(job) is True

    def test_valid_record_controls_retry_policy(self, env, monkeypatch):
        root, _calls = env
        directory, job = _job(root, "n9-known", ["❌ CUDA out of memory"])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: True)
        identity = app_module._prepare_worker_failure("n9-known", job)
        app_module.worker_failure.write_record(
            directory,
            identity,
            app_module.worker_failure.WorkerFailure(
                "backend_unavailable", "whisper", "mlx", ("mlx",)),
        )
        job["status"] = "failed"
        job["worker_failure"] = app_module._read_worker_failure(job, identity)

        assert app_module._should_auto_retry(job) is False

    def test_incomplete_output_retries_once(self, env, monkeypatch):
        root, _calls = env
        directory, job = _job(root, "n9-render", [])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: True)
        identity = app_module._prepare_worker_failure("n9-render", job)
        app_module.worker_failure.write_record(
            directory,
            identity,
            app_module.worker_failure.WorkerFailure("incomplete_output", "render", "ffmpeg"),
        )
        job["status"] = "failed"
        job["worker_failure"] = app_module._read_worker_failure(job, identity)

        assert app_module._should_auto_retry(job) is True

    def test_new_execution_clears_the_previous_record(self, env, monkeypatch):
        root, _calls = env
        directory, job = _job(root, "n9-next", [])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: True)
        first = app_module._prepare_worker_failure("n9-next", job)
        app_module.worker_failure.write_record(
            directory, first,
            app_module.worker_failure.WorkerFailure("service_unavailable", "clip_selection", "ollama"),
        )

        second = app_module._prepare_worker_failure("n9-next", job)

        assert second.execution_generation == first.execution_generation + 1
        assert app_module.worker_failure.read_record(directory, first) is None
        assert job["env"]["OPENSHORTS_WORKER_FAILURE_EXECUTION_GENERATION"] == "2"

    def test_run_job_reads_only_a_matching_record(self, env, monkeypatch):
        root, _calls = env
        directory, job = _job(root, "n9-worker", [])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: True)

        class Output:
            def readline(self):
                return b""

            def close(self):
                return None

        class Process:
            stdout = Output()
            returncode = 1

            def poll(self):
                return self.returncode

        def fake_popen(*_args, **kwargs):
            env = kwargs["env"]
            identity = app_module.worker_failure.FailureIdentity(
                env["OPENSHORTS_WORKER_FAILURE_JOB_ID"],
                int(env["OPENSHORTS_WORKER_FAILURE_CONFIGURATION_REVISION"]),
                int(env["OPENSHORTS_WORKER_FAILURE_EXECUTION_GENERATION"]),
            )
            app_module.worker_failure.write_record(
                directory, identity,
                app_module.worker_failure.WorkerFailure(
                    "service_unavailable", "clip_selection", "ollama", ("ollama",)),
            )
            return Process()

        monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)
        asyncio.run(app_module.run_job("n9-worker", job))

        assert job["status"] == "failed"
        assert job["worker_failure"]["code"] == "service_unavailable"
        assert app_module._should_auto_retry(job) is True

    def test_terminal_record_overrides_a_zero_exit_status(self, env, monkeypatch):
        root, _calls = env
        directory, job = _job(root, "n9-zero-exit", [])
        monkeypatch.setattr(app_module.worker_failure, "records_enabled", lambda: True)

        class Output:
            def readline(self):
                return b""

            def close(self):
                return None

        class Process:
            stdout = Output()
            returncode = 0

            def poll(self):
                return self.returncode

        def fake_popen(*_args, **kwargs):
            env = kwargs["env"]
            identity = app_module.worker_failure.FailureIdentity(
                env["OPENSHORTS_WORKER_FAILURE_JOB_ID"],
                int(env["OPENSHORTS_WORKER_FAILURE_CONFIGURATION_REVISION"]),
                int(env["OPENSHORTS_WORKER_FAILURE_EXECUTION_GENERATION"]),
            )
            app_module.worker_failure.write_record(
                directory, identity,
                app_module.worker_failure.WorkerFailure(
                    "service_unavailable", "clip_selection", "ollama", ("ollama",)),
            )
            return Process()

        monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)
        asyncio.run(app_module.run_job("n9-zero-exit", job))

        assert job["status"] == "failed"
        assert app_module._should_auto_retry(job) is True


class TestResumeMidRender:
    def test_manifest_with_metadata_is_resumed_not_recovered(self, env, monkeypatch):
        root, calls = env
        d = root / "j5"
        d.mkdir()
        (d / "My_Video.mp4").write_bytes(b"src")
        config = app_module.job_state.seal_job_config({
            "schema_version": 1,
            "job_id": "j5",
            "configuration_revision": 1,
            "project_dir": str(d),
            "source": {
                "kind": "upload",
                "retained_relative_path": "My_Video.mp4",
            },
            "pipeline": {},
            "environment": {},
            "checkpointing_enabled": True,
        })
        app_module.job_state.write_job_config(str(d), config)
        app_module.job_state.write_job_state(str(d), {
            "schema_version": 1,
            "job_id": "j5",
            "configuration_revision": 1,
            "execution_generation": 0,
            "status": "processing",
        })
        app_module.job_state.write_resume_manifest(str(d), {
            "schema_version": 1,
            "job_id": "j5",
            "configuration_revision": 1,
            "priority": 1,
            "attempts": 0,
        })
        (d / "j5_metadata.json").write_text(json.dumps({"shorts": [{}]}))
        app_module._recover_jobs_from_disk()
        assert "j5" not in app_module.jobs                    # not "completed"
        app_module._resume_interrupted_jobs()
        assert app_module.jobs["j5"]["status"] == "queued"
        assert calls["enqueued"] == [("j5", 1)]


def test_terminal_outbox_stops_after_retry_budget(env, monkeypatch):
    root, _calls = env
    d, _job_record = _job(root, "outbox-budget", [])
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": "outbox-budget",
        "configuration_revision": 1,
        "execution_generation": 1,
        "status": "processing",
    })
    app_module.job_state.commit_terminal_state(
        str(d),
        1,
        "failed",
        {},
        [],
        {"actions": {"archive": True}},
    )
    monkeypatch.setattr(app_module, "TERMINAL_ACTION_MAX_ATTEMPTS", 2)
    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("archive unavailable")

    async def run():
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await app_module._run_terminal_effect(
                    "outbox-budget", "archive", fail,
                )
        await app_module._run_terminal_effect(
            "outbox-budget", "archive", fail,
        )

    asyncio.run(run())
    record = app_module.job_state.load_job_state(str(d))[
        "terminal_actions"
    ]["archive"]
    assert calls == 2
    assert record["attempts"] == 2
    assert record["state"] == "permanent_failure"


def test_ambiguous_terminal_failure_requires_manual_review(env):
    root, _calls = env
    d, _job_record = _job(root, "ambiguous-outbox", [])
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": "ambiguous-outbox",
        "configuration_revision": 1,
        "execution_generation": 1,
        "status": "processing",
    })
    app_module.job_state.commit_terminal_state(
        str(d),
        1,
        "failed",
        {},
        [],
        {"actions": {"alert": True}},
    )
    calls = 0

    async def uncertain_delivery():
        nonlocal calls
        calls += 1
        raise RuntimeError("connection closed after request body")

    with pytest.raises(RuntimeError):
        asyncio.run(app_module._run_terminal_effect(
            "ambiguous-outbox",
            "alert",
            uncertain_delivery,
            ambiguous=True,
        ))
    record = app_module.job_state.load_job_state(str(d))[
        "terminal_actions"
    ]["alert"]
    assert calls == 1
    assert record["state"] == "manual_review"


def test_disabled_checkpoint_config_does_not_resolve_model_digests(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(app_module.llm_backend, "active", lambda: True)
    monkeypatch.setattr(app_module.vision_backend, "active", lambda: True)
    monkeypatch.setattr(
        app_module.llm_backend,
        "resolved_descriptor",
        lambda: (_ for _ in ()).throw(AssertionError("network preflight")),
    )
    monkeypatch.setattr(
        app_module.vision_backend,
        "resolved_descriptor",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("network preflight")
        ),
    )
    monkeypatch.setattr(
        app_module.llm_backend,
        "configured_descriptor",
        lambda: {
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "requested_model": "qwen",
            "immutable_model_revision": None,
            "immutable_revision_available": False,
        },
    )
    monkeypatch.setattr(
        app_module.vision_backend,
        "configured_descriptor",
        lambda _seconds: {
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "model": "qwen-vl",
            "digest": None,
            "immutable_revision_available": False,
        },
    )
    config = app_module._resolved_job_config(
        "disabled-preflight",
        str(tmp_path),
        {"kind": "upload", "retained_relative_path": "source.mp4"},
        {},
        "auto",
        [],
        auto_hook=False,
        auto_hook_style=None,
        captions=True,
        target_clips=3,
        clip_min_seconds=20,
        clip_max_seconds=60,
        checkpointing_enabled=False,
    )
    assert config["checkpointing_enabled"] is False
    assert config["text_selection"]["immutable_revision_available"] is False
    assert config["vision_selection"]["immutable_revision_available"] is False


def test_live_checkpoint_kill_switch_overrides_stored_environment(monkeypatch):
    config = {
        "checkpointing_enabled": True,
        "environment": {
            "PIPELINE_CHECKPOINTS_ENABLED": "true",
            "LLM_MODEL": "qwen2.5:7b",
        },
    }
    monkeypatch.setattr(app_module, "PIPELINE_CHECKPOINTS_ENABLED", False)
    disabled = app_module.build_job_child_env(config, {})
    assert disabled["PIPELINE_CHECKPOINTS_ENABLED"] == "false"
    assert disabled["LLM_MODEL"] == "qwen2.5:7b"

    config["checkpointing_enabled"] = False
    monkeypatch.setattr(app_module, "PIPELINE_CHECKPOINTS_ENABLED", True)
    still_disabled = app_module.build_job_child_env(config, {})
    assert still_disabled["PIPELINE_CHECKPOINTS_ENABLED"] == "false"

    config["checkpointing_enabled"] = True
    enabled = app_module.build_job_child_env(config, {})
    assert enabled["PIPELINE_CHECKPOINTS_ENABLED"] == "true"


def test_corrupt_durable_outbox_never_runs_legacy_callback(env):
    root, _calls = env
    d = root / "corrupt-outbox"
    d.mkdir()
    (d / ".job-state.json").write_text("{not-json")
    app_module.jobs["corrupt-outbox"] = {
        "output_dir": str(d),
        "status": "failed",
        "logs": [],
    }
    called = False

    async def effect():
        nonlocal called
        called = True

    with pytest.raises(app_module.job_state.JobStateError):
        asyncio.run(app_module._run_terminal_effect(
            "corrupt-outbox", "archive", effect,
        ))
    assert called is False


def test_new_execution_generation_drops_old_live_plan(env):
    root, _calls = env
    d = root / "next-generation"
    d.mkdir()
    (d / "My_Video.mp4").write_bytes(b"src")
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": "next-generation",
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "upload",
            "retained_relative_path": "My_Video.mp4",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
    })
    app_module.job_state.write_job_config(str(d), config)
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": "next-generation",
        "configuration_revision": 1,
        "execution_generation": 1,
        "status": "queued",
        "live_plan": {
            "execution_generation": 1,
            "metadata_filename": "next-generation_metadata.json",
            "plan_hash": "a" * 64,
            "count": 2,
        },
    })
    app_module.job_state.write_resume_manifest(str(d), {
        "schema_version": 1,
        "job_id": "next-generation",
        "configuration_revision": 1,
        "attempts": 0,
    })
    job = {
        "output_dir": str(d),
        "runtime_secrets": {},
        "status": "queued",
        "logs": [],
    }

    lease, generation = app_module.claim_job_execution(
        "next-generation", job,
    )
    try:
        state = app_module.job_state.load_job_state(str(d))
        assert generation == 2
        assert state["execution_generation"] == 2
        assert state["status"] == "processing"
        assert "live_plan" not in state
    finally:
        lease.release()


def test_resume_waits_for_private_source_credentials_when_retained_copy_is_missing(
        env, monkeypatch):
    root, calls = env
    job_id = "private-source"
    d = root / job_id
    d.mkdir()
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "url",
            "retained_relative_path": "source.mp4",
            "canonical_public_url": "https://www.youtube.com/watch?v=private",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
        "credential_requirements": {
            "source_authorization": True,
        },
    })
    app_module.job_state.write_job_config(str(d), config)
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "execution_generation": 0,
        "status": "processing",
    })
    app_module.job_state.write_resume_manifest(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "attempts": 0,
    })
    monkeypatch.setattr(
        app_module.job_state,
        "load_resume_secret_keyring",
        lambda: {"active_key_id": None, "keys": {}},
    )
    monkeypatch.setattr(
        app_module.job_state, "load_resume_secrets", lambda *_a, **_k: None,
    )

    app_module._resume_interrupted_jobs()

    state = app_module.job_state.load_job_state(str(d))
    assert state["status"] == "waiting_for_credentials"
    assert job_id not in app_module.jobs
    assert calls["enqueued"] == []


def test_waiting_for_source_does_not_consume_resume_attempts(env, monkeypatch):
    root, calls = env
    job_id = "missing-upload-source"
    d = root / job_id
    d.mkdir()
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "upload",
            "retained_relative_path": "source.mp4",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
    })
    app_module.job_state.write_job_config(str(d), config)
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "execution_generation": 0,
        "status": "processing",
    })
    app_module.job_state.write_resume_manifest(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "attempts": 1,
    })
    monkeypatch.setattr(
        app_module.job_state,
        "load_resume_secret_keyring",
        lambda: {"active_key_id": None, "keys": {}},
    )
    monkeypatch.setattr(
        app_module.job_state, "load_resume_secrets", lambda *_a, **_k: None,
    )

    app_module._resume_interrupted_jobs()
    app_module._resume_interrupted_jobs()

    state = app_module.job_state.load_job_state(str(d))
    manifest = app_module.job_state.load_resume_manifest(str(d))
    assert state["status"] == "waiting_for_source"
    assert manifest["attempts"] == 1
    assert job_id not in app_module.jobs
    assert calls["enqueued"] == []


def test_unknown_resume_route_requires_any_potential_gemini_key():
    assert app_module._selection_credentials_missing(
        {"gemini_silent": True},
        {},
        None,
    )
    assert not app_module._selection_credentials_missing(
        {"gemini_silent": True},
        {"GEMINI_API_KEY": "present"},
        None,
    )


def test_completed_asr_requires_only_its_selection_route_key():
    assert not app_module._selection_credentials_missing(
        {"gemini_silent": True},
        {},
        "speech",
    )
    assert app_module._selection_credentials_missing(
        {"gemini_speech": True},
        {},
        "speech",
    )
    assert app_module._selection_credentials_missing(
        {"gemini_silent": True},
        {},
        "silent",
    )


def test_completed_clip_plan_needs_no_selection_provider_key():
    assert not app_module._selection_credentials_missing(
        {
            "gemini_speech": True,
            "gemini_silent": True,
            "llm_authentication": True,
        },
        {},
        "complete",
    )


def test_resume_does_not_require_private_source_credentials_with_retained_copy(
        env, monkeypatch):
    root, calls = env
    job_id = "private-source-retained"
    d = root / job_id
    d.mkdir()
    (d / "source.mp4").write_bytes(b"retained")
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "url",
            "retained_relative_path": "source.mp4",
            "canonical_public_url": "https://www.youtube.com/watch?v=private",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
        "credential_requirements": {
            "source_authorization": True,
        },
    })
    app_module.job_state.write_job_config(str(d), config)
    identity = {
        "sha256": app_module.checkpoint_store.sha256_file(
            str(d / "source.mp4")
        ),
        "size": (d / "source.mp4").stat().st_size,
        "duration_ms": 1000,
    }
    store = app_module.checkpoint_store.CheckpointStore(
        str(d), config, True, 1,
    )

    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(
        app_module.checkpoint_store.subprocess,
        "run",
        lambda *_a, **_k: Probe(),
    )
    assert store.save_source(
        app_module.checkpoint_store.expected_source_request_key(config),
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
    app_module.job_state.write_job_state(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "execution_generation": 0,
        "status": "processing",
    })
    app_module.job_state.write_resume_manifest(str(d), {
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "attempts": 0,
    })
    monkeypatch.setattr(
        app_module.job_state,
        "load_resume_secret_keyring",
        lambda: {"active_key_id": None, "keys": {}},
    )
    monkeypatch.setattr(
        app_module.job_state, "load_resume_secrets", lambda *_a, **_k: None,
    )

    app_module._resume_interrupted_jobs()

    assert app_module.jobs[job_id]["status"] == "queued"
    assert calls["enqueued"] == [(job_id, 2)]


def test_uncheckpointed_retained_url_is_redownloaded(env, monkeypatch):
    root, _calls = env
    # The operational kill switch disables downstream checkpoint reuse, but it
    # must not make an unverified partial URL download safe to consume.
    monkeypatch.setattr(app_module, "PIPELINE_CHECKPOINTS_ENABLED", False)
    job_id = "partial-url-source"
    d = root / job_id
    d.mkdir()
    (d / "source.mp4").write_bytes(b"possibly-partial")
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "url",
            "retained_relative_path": "source.mp4",
            "canonical_public_url": "https://www.youtube.com/watch?v=source",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
    })

    command = app_module.build_job_command(config)

    assert "-u" in command
    source_flag = command.index("-u", command.index("main.py") + 1)
    assert command[source_flag + 1] == "https://www.youtube.com/watch?v=source"
    assert "-i" not in command


def test_completed_checkpoint_does_not_require_lost_handover_transcript(
        env, monkeypatch):
    root, _calls = env
    job_id = "completed-precomputed"
    d = root / job_id
    d.mkdir()
    (d / "source.mp4").write_bytes(b"source")
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "thumbnail_session",
            "retained_relative_path": "source.mp4",
            "transcript_relative_path": "source_transcript.json",
            "transcript_sha256": "a" * 64,
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
    })
    monkeypatch.setattr(
        app_module, "_retained_source_is_usable", lambda _config: True,
    )
    monkeypatch.setattr(
        app_module, "_checkpointed_selection_route",
        lambda _config: "complete",
    )

    command = app_module.build_job_command(config)

    assert "-i" in command
    assert "--transcript" not in command


def test_corrupt_retained_source_is_not_reused_under_original_url_config(
        env, monkeypatch):
    root, _calls = env
    job_id = "changed-source"
    d = root / job_id
    d.mkdir()
    source = d / "source.mp4"
    source.write_bytes(b"original-source")
    config = app_module.job_state.seal_job_config({
        "schema_version": 1,
        "job_id": job_id,
        "configuration_revision": 1,
        "project_dir": str(d),
        "source": {
            "kind": "url",
            "retained_relative_path": "source.mp4",
            "canonical_public_url": "https://www.youtube.com/watch?v=source",
        },
        "pipeline": {},
        "environment": {},
        "checkpointing_enabled": True,
    })
    identity = {
        "sha256": app_module.checkpoint_store.sha256_file(str(source)),
        "size": source.stat().st_size,
        "duration_ms": 1000,
    }
    store = app_module.checkpoint_store.CheckpointStore(
        str(d), config, True, 1,
    )

    class Probe:
        returncode = 0
        stdout = "1.0\n"

    monkeypatch.setattr(
        app_module.checkpoint_store.subprocess,
        "run",
        lambda *_a, **_k: Probe(),
    )
    assert store.save_source(
        app_module.checkpoint_store.expected_source_request_key(config),
        {
            "source_fingerprint": identity,
            "retained_relative_path": "source.mp4",
            "artifact": {
                "relative_path": "source.mp4",
                "size": identity["size"],
                "sha256": identity["sha256"],
                "duration_ms": 1000,
                "probe": {"playable": True},
            },
            "probe": {"duration": 1.0, "width": 1920, "height": 1080},
        },
        {},
    )
    monkeypatch.setattr(app_module, "PIPELINE_CHECKPOINTS_ENABLED", True)
    source.write_bytes(b"different-source")

    command = app_module.build_job_command(config)

    assert "-i" not in command
    source_flag = command.index("-u", command.index("main.py") + 1)
    assert command[source_flag + 1] == "https://www.youtube.com/watch?v=source"


def test_terminal_companion_chain_is_bounded_and_exact():
    assert app_module._terminal_companion_names(
        "wm_subtitled_12_hooked_11_clip-deadbeef.mp4"
    ) == [
        "subtitled_12_hooked_11_clip-deadbeef.mp4",
        "hooked_11_clip-deadbeef.mp4",
        "clip-deadbeef.mp4",
    ]
    assert app_module._terminal_companion_names(
        "subtitled_12_clip-deadbeef.mp4"
    ) == ["clip-deadbeef.mp4"]
    assert app_module._terminal_companion_names(
        "unrelated-file.mp4"
    ) == []


class TestQueueEstimate:
    def test_free_slot_starts_now(self):
        assert app_module.queue_estimate(ahead=0, running=3, slots=8, typical_seconds=240) == 15

    def test_grows_with_the_line(self):
        a = app_module.queue_estimate(ahead=8, running=8, slots=8, typical_seconds=240)
        b = app_module.queue_estimate(ahead=16, running=8, slots=8, typical_seconds=240)
        assert 30 <= a < b

    def test_snapshot_positions(self, env, monkeypatch):
        import asyncio as _a
        q = _a.PriorityQueue()
        for i, (prio, jid) in enumerate([(1, "a"), (2, "b"), (1, "c")]):
            q.put_nowait((prio, i, jid))
            app_module.jobs[jid] = {"status": "queued", "logs": []}
        monkeypatch.setattr(app_module, "job_queue", q)
        monkeypatch.setattr(app_module, "_admitting", {"z"})
        app_module.jobs["z"] = {"status": "queued", "logs": []}
        assert app_module.queue_snapshot("z")["position"] == 1
        assert app_module.queue_snapshot("a")["position"] == 2
        assert app_module.queue_snapshot("c")["position"] == 3
        assert app_module.queue_snapshot("b")["position"] == 4
        app_module.jobs["a"]["status"] = "processing"
        assert app_module.queue_snapshot("a") is None


@pytest.mark.parametrize("value,expected", [(None, 10), ("0", 0), ("5", 5), ("40", 19),
                                            ("-3", 0), ("x", 10)])
def test_job_processes_run_niced(monkeypatch, value, expected):
    """Jobs are batch work: under CPU pressure the API (and editor renders,
    which run in it) go first. JOB_NICE=0 turns it off."""
    if value is None:
        monkeypatch.delenv("JOB_NICE", raising=False)
    else:
        monkeypatch.setenv("JOB_NICE", value)
    assert app_module._job_nice() == expected
