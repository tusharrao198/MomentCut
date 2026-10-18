"""The hourly sweeps must never delete a job that is still queued or running.

22-sep-2026: with the queue backed up, free jobs waited over an hour and the
age-only sweep removed their dir (and source upload) under the worker. The
user got no clips and the minute reservation stayed held for 3 hours.
"""
import asyncio
import os
import time
import types

import pytest

app_module = pytest.importorskip("app")

OLD = 2 * 3600  # older than JOB_RETENTION_SECONDS in cloud (1 h)


@pytest.fixture
def disk(tmp_path, monkeypatch):
    out = tmp_path / "output"
    up = tmp_path / "uploads"
    out.mkdir()
    up.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up))
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(out / "thumbnails"))
    monkeypatch.setattr(app_module, "JOB_RETENTION_SECONDS", 3600)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_job_activity_counts", {})
    monkeypatch.setattr(app_module, "_postiz_publish_locks", {})
    monkeypatch.setattr(app_module, "_clip_thumbnail_locks", {})
    monkeypatch.setattr(app_module, "jobs", {})
    return out, up


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


def _job_dir(out, job_id, manifest=False, age=OLD):
    d = out / job_id
    d.mkdir()
    if manifest:
        m = d / app_module._RESUME_FILE
        m.write_text("{}")
        _age(m, age)
    _age(d, age)
    return d


def _upload(up, job_id, age=OLD):
    f = up / f"{job_id}_video.mp4"
    f.write_bytes(b"x")
    _age(f, age)
    return f


class TestJobDirSweep:
    def test_finished_old_job_is_purged(self, disk):
        out, _ = disk
        d = _job_dir(out, "done")
        app_module.jobs["done"] = {"status": "completed"}
        app_module._sweep_expired_job_dirs(time.time())
        assert not d.exists()
        assert "done" not in app_module.jobs

    def test_queued_old_job_survives(self, disk):
        out, _ = disk
        d = _job_dir(out, "waiting")
        app_module.jobs["waiting"] = {"status": "queued"}
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists() and "waiting" in app_module.jobs

    def test_running_old_job_survives(self, disk):
        out, _ = disk
        d = _job_dir(out, "busy")
        app_module.jobs["busy"] = {"status": "processing"}
        app_module._running_jobs.add("busy")
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists() and "busy" in app_module.jobs

    def test_manifest_of_other_instance_survives(self, disk):
        out, _ = disk
        # Not in this instance's memory, but a fresh manifest says it is live.
        d = _job_dir(out, "handover", manifest=True, age=30)
        _age(d, OLD)
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists()

    def test_abandoned_manifest_does_not_pin_forever(self, disk):
        out, _ = disk
        d = _job_dir(out, "ghost", manifest=True,
                     age=app_module.ACTIVE_JOB_MAX_SECONDS + 60)
        app_module._sweep_expired_job_dirs(time.time())
        assert not d.exists()

    def test_size_cap_skips_live_jobs(self, disk, monkeypatch):
        out, _ = disk
        live = _job_dir(out, "live")
        (live / "clip.mp4").write_bytes(b"x" * 2048)
        app_module.jobs["live"] = {"status": "queued"}
        dead = _job_dir(out, "dead")
        (dead / "clip.mp4").write_bytes(b"x" * 2048)
        monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 1 / 1024 ** 3)  # 1 byte
        app_module._enforce_output_size_cap()
        assert live.exists() and not dead.exists()

    def test_secondary_activity_lease_pins_completed_job(self, disk):
        out, _ = disk
        job = _job_dir(out, "postiz-upload")
        app_module.jobs["postiz-upload"] = {"status": "completed"}

        async def sweep_while_pinned():
            async with app_module._job_activity_lease("postiz-upload"):
                app_module._sweep_expired_job_dirs(time.time())
                assert job.exists()
                assert app_module._job_is_active("postiz-upload") is True

        asyncio.run(sweep_while_pinned())
        assert app_module._job_is_active("postiz-upload") is False
        app_module._sweep_expired_job_dirs(time.time())
        assert not job.exists()

    def test_held_secondary_lock_pins_job_and_idle_locks_are_forgotten(self, disk):
        out, _ = disk
        job = _job_dir(out, "thumbnail-state")
        app_module.jobs["thumbnail-state"] = {"status": "completed"}
        async def sweep_around_lock():
            lock = await app_module._clip_thumbnail_lock("thumbnail-state", 0)
            await lock.acquire()
            try:
                app_module._sweep_expired_job_dirs(time.time())
                assert job.exists()
            finally:
                lock.release()

        asyncio.run(sweep_around_lock())
        app_module._sweep_expired_job_dirs(time.time())

        assert not job.exists()
        assert not any(key[0] == "thumbnail-state"
                       for key in app_module._clip_thumbnail_locks)

    def test_thumbnail_state_recovery_is_pinned_before_it_takes_its_lock(
        self, disk, monkeypatch
    ):
        """A sweep cannot win the brief gap before the handler's clip lock."""
        seen = {}

        async def fake_state_read(job_id, clip_index, request):
            seen["active"] = app_module._job_is_active(job_id)
            return {"success": True}

        monkeypatch.setattr(app_module, "_get_clip_thumbnail_state", fake_state_read)
        result = asyncio.run(
            app_module.get_clip_thumbnail_state("state-read", 0, object())
        )

        assert result == {"success": True}
        assert seen["active"] is True
        assert app_module._job_is_active("state-read") is False


class TestUploadSweep:
    def test_source_of_queued_job_survives(self, disk):
        _, up = disk
        f = _upload(up, "waiting")
        app_module.jobs["waiting"] = {"status": "queued"}
        app_module._sweep_expired_uploads(time.time())
        assert f.exists()

    def test_source_of_finished_job_is_removed(self, disk):
        _, up = disk
        f = _upload(up, "done")
        app_module.jobs["done"] = {"status": "failed"}
        app_module._sweep_expired_uploads(time.time())
        assert not f.exists()

    def test_size_cap_skips_live_sources(self, disk, monkeypatch):
        _, up = disk
        live = _upload(up, "live")
        app_module._running_jobs.add("live")
        dead = _upload(up, "dead")
        monkeypatch.setattr(app_module, "UPLOADS_MAX_GB", 1 / 1024 ** 3)
        app_module._enforce_uploads_size_cap()
        assert live.exists() and not dead.exists()


class TestSettleWithoutEntry:
    def test_reservation_released_when_entry_vanished(self, monkeypatch):
        calls = []

        class FakeMetering:
            @staticmethod
            async def commit_reservation(rid):
                calls.append(("commit", rid))

            @staticmethod
            async def release_reservation(rid):
                calls.append(("release", rid))

        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
        monkeypatch.setattr(app_module, "cloud", types.SimpleNamespace(metering=FakeMetering), raising=False)
        monkeypatch.setattr(app_module, "jobs", {})
        job = {"status": "processing", "reservation_id": "res-1"}
        asyncio.run(app_module._settle_reservation("gone", job))
        assert calls == [("release", "res-1")]
