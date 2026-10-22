import json

import pytest

import worker_failure
from accelerator_runtime import AcceleratorConfigurationError


def identity():
    return worker_failure.FailureIdentity("job-1", 1, 2)


def test_auto_mode_follows_the_apple_feature_gate():
    assert worker_failure.records_enabled("auto", apple_acceleration_active=True) is True
    assert worker_failure.records_enabled("auto", apple_acceleration_active=False) is False
    assert worker_failure.records_enabled("enabled", apple_acceleration_active=False) is True
    assert worker_failure.records_enabled("disabled", apple_acceleration_active=True) is False


def test_write_and_read_matching_record(tmp_path):
    failure = worker_failure.WorkerFailure(
        "resource_exhausted", "whisper", "mlx", ("mlx", "cpu"))

    written = worker_failure.write_record(tmp_path, identity(), failure)

    assert worker_failure.read_record(tmp_path, identity()) == written
    assert worker_failure.should_auto_retry(written) is True


def test_rejects_wrong_execution_identity(tmp_path):
    worker_failure.write_record(
        tmp_path, identity(), worker_failure.WorkerFailure("backend_unavailable", "render"))

    assert worker_failure.read_record(
        tmp_path, worker_failure.FailureIdentity("job-1", 1, 3)) is None


def test_rejects_malformed_or_oversized_record(tmp_path):
    path = worker_failure.failure_path(tmp_path)
    path.write_text(json.dumps({"code": "resource_exhausted"}))
    assert worker_failure.read_record(tmp_path, identity()) is None

    path.write_bytes(b"x" * (worker_failure.MAX_RECORD_BYTES + 1))
    assert worker_failure.read_record(tmp_path, identity()) is None


def test_reporter_classifies_only_typed_configuration_errors(tmp_path):
    reporter = worker_failure.FailureReporter(tmp_path, identity())
    reporter.set_context("whisper", "mlx", ("mlx",))

    record = reporter.record(AcceleratorConfigurationError("bad configuration"))

    assert record["code"] == "backend_unavailable"
    assert record["stage"] == "whisper"
    assert record["backend"] == "mlx"


def test_reporter_uses_explicit_worker_failure_and_writes_once(tmp_path):
    reporter = worker_failure.FailureReporter(tmp_path, identity())
    expected = worker_failure.WorkerFailure("incomplete_output", "render", "ffmpeg")

    assert reporter.record(expected)["code"] == "incomplete_output"
    assert reporter.record(RuntimeError("later error"))["code"] == "incomplete_output"


def test_uncaught_cancellation_never_writes_a_terminal_record(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_failure.sys, "excepthook", lambda *_args: None)
    reporter = worker_failure.FailureReporter(tmp_path, identity())
    reporter.install_uncaught_exception_hook()

    worker_failure.sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert worker_failure.read_record(tmp_path, identity()) is None


def test_clear_record_does_not_fail_when_the_path_is_not_a_file(tmp_path):
    worker_failure.failure_path(tmp_path).mkdir()

    worker_failure.clear_record(tmp_path)

    assert worker_failure.failure_path(tmp_path).is_dir()


def test_reporter_ignores_partial_environment(tmp_path):
    assert worker_failure.reporter_from_environment(tmp_path, {}) is None
    assert worker_failure.reporter_from_environment(tmp_path, {
        "OPENSHORTS_WORKER_FAILURE_JOB_ID": "job-1",
    }) is None


def test_record_accepts_portable_whisper_backend():
    record = worker_failure.WorkerFailure(
        "backend_unavailable", "whisper", "faster-whisper", ("faster-whisper",))

    assert record.backend == "faster-whisper"


def test_record_rejects_unallowlisted_values():
    with pytest.raises(ValueError):
        worker_failure.WorkerFailure("unsafe-detail")
