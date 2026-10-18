import asyncio
import base64
import hashlib
import io
import json
import uuid

import httpx
import pytest
from PIL import Image

import clip_thumbnail_image

app_module = pytest.importorskip("app")

JOB_ID = "12345678-1234-1234-1234-123456789abc"
ORIGIN = "http://localhost:5175"


def _jpeg_bytes():
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    output = io.BytesIO()
    image.save(output, "JPEG")
    return output.getvalue()


def _different_jpeg_bytes():
    image = Image.new("RGB", (8, 8), color=(90, 80, 70))
    output = io.BytesIO()
    image.save(output, "JPEG")
    return output.getvalue()


def _request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await getattr(client, method)(path, **kwargs)
    return asyncio.run(send())


@pytest.fixture()
def thumbnail_job(tmp_path, monkeypatch):
    output = tmp_path / "output"
    job_dir = output / JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "source_metadata.json").write_text(json.dumps({
        "transcript": {
            "segments": [{
                "words": [
                    {"word": "This", "start": 0, "end": 1},
                    {"word": "moment", "start": 1, "end": 2},
                ]
            }]
        }
    }), encoding="utf-8")
    (job_dir / "clip.mp4").write_bytes(b"rendered-video")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(output))
    monkeypatch.setattr(app_module, "jobs", {
        JOB_ID: {
            "result": {
                "clips": [{
                    "start": 0,
                    "end": 2,
                    "video_url": f"/videos/{JOB_ID}/clip.mp4",
                    "video_title_for_youtube_short": "A valid YouTube title",
                    "video_description_for_tiktok": "A description",
                    "thumbnail_brief": {"thumbnail_text": "THE MOMENT"},
                }],
            },
        },
    })
    monkeypatch.setattr(app_module, "_clip_thumbnail_locks", {})
    monkeypatch.setattr(app_module, "_clip_thumbnail_generation_gate", asyncio.BoundedSemaphore(1))
    monkeypatch.setattr(app_module, "_clip_thumbnail_generation_gate_guard", asyncio.Lock())
    monkeypatch.setattr(app_module, "_clip_thumbnail_active_request", None)
    monkeypatch.setenv("GEMINI_API_KEY", "not-exposed")
    monkeypatch.setenv("CLIP_AI_THUMBNAILS_ENABLED", "1")
    return job_dir


def _generate(request_id=None, origin=ORIGIN, count=1):
    return _request(
        "post",
        "/api/clip-thumbnail/generate",
        headers={"Origin": origin},
        json={
            "job_id": JOB_ID,
            "clip_index": 0,
            "instructions": "make it cinematic",
            "count": count,
            "request_id": request_id or str(uuid.uuid4()),
        },
    )


def test_generate_requires_local_dashboard_origin(thumbnail_job):
    response = _generate(origin="https://attacker.example")
    assert response.status_code == 403
    assert "local OpenShorts dashboard" in response.json()["detail"]


def test_thumbnail_state_allows_originless_same_origin_get(thumbnail_job):
    # Browsers commonly omit Origin on same-origin GET requests. The dashboard
    # proxies this read-only endpoint through Vite, while all thumbnail writes
    # still require an explicit trusted local Origin.
    response = _request("get", f"/api/clip-thumbnail/{JOB_ID}/0")
    assert response.status_code == 200
    assert response.json()["youtube"]["title"] == "A valid YouTube title"


def test_generate_is_idempotent_and_selected_candidate_is_confined(thumbnail_job, monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module,
        "_make_clip_reference_frame",
        lambda *_args: (None, None),
    )
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: calls.append(True) or _jpeg_bytes(),
    )
    request_id = str(uuid.uuid4())

    first = _generate(request_id)
    second = _generate(request_id)
    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == [True]

    candidate = first.json()["candidates"][0]
    selected = _request(
        "post",
        f"/api/clip-thumbnail/{JOB_ID}/0/select",
        headers={"Origin": ORIGIN},
        json={"candidate_id": candidate["id"]},
    )
    assert selected.status_code == 200
    assert selected.json()["selectedCandidateId"] == candidate["id"]

    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, digest = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    thumbnail = app_module._selected_ai_thumbnail(JOB_ID, 0, filename, digest)
    assert thumbnail["content_type"] == "image/jpeg"
    assert thumbnail["filename"].endswith(".jpg")
    assert (thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE).exists()


def test_unknown_image_request_is_not_retried_automatically(thumbnail_job, monkeypatch):
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            clip_thumbnail_image.ClipThumbnailRequestUnknown("request did not complete")
        ),
    )
    request_id = str(uuid.uuid4())

    response = _generate(request_id)
    assert response.status_code == 502
    second = _generate(request_id)
    assert second.status_code == 200
    assert second.json()["batchState"] == "unknown"


def test_known_gemini_quota_rejection_is_retryable(thumbnail_job, monkeypatch):
    """A 429 response is a confirmed rejection, not an uncertain paid request."""
    class QuotaError(Exception):
        code = 429

    class Model:
        def generate_content(self, **_kwargs):
            raise QuotaError()

    class Client:
        models = Model()

    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(clip_thumbnail_image.genai, "Client", lambda **_kwargs: Client())

    response = _generate()

    assert response.status_code == 502
    assert "no Gemini API free tier" in response.json()["detail"]
    retry = _generate()
    assert retry.status_code == 502
    assert "batchState" not in retry.json()


def test_requested_candidate_count_makes_one_call_per_candidate(thumbnail_job, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: calls.append(True) or _jpeg_bytes(),
    )

    response = _generate(count=3)

    assert response.status_code == 200
    assert len(calls) == 3
    assert len(response.json()["candidates"]) == 3


def test_partial_candidate_batch_keeps_completed_candidates(thumbnail_job, monkeypatch):
    calls = []

    def generate(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise clip_thumbnail_image.ClipThumbnailImageError("Gemini quota is exhausted.")
        return _jpeg_bytes()

    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(clip_thumbnail_image, "generate_clip_thumbnail", generate)

    response = _generate(count=3)

    assert response.status_code == 200
    assert len(calls) == 2
    assert len(response.json()["candidates"]) == 1
    assert response.json()["batchState"] == "failed"
    assert "Generated 1 of 3 candidates" in response.json()["message"]


def test_orphaned_running_batch_is_recovered_by_state_read_before_a_new_request(
    thumbnail_job, monkeypatch
):
    calls = []
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: calls.append(True) or _jpeg_bytes(),
    )
    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, fingerprint = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    state_path = thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    state_path.write_text(json.dumps({
        "versions": {
            f"{filename}:{fingerprint}": {
                "filename": filename,
                "sha256": fingerprint,
                "selected_candidate_id": None,
                "batches": {
                    str(uuid.uuid4()): {
                        "state": "running",
                        "requested_count": 1,
                        "started_at": "2026-10-18T00:00:00+00:00",
                        "candidates": [],
                    },
                },
            },
        },
    }), encoding="utf-8")

    recovered = _request("get", f"/api/clip-thumbnail/{JOB_ID}/0")
    assert recovered.status_code == 200
    assert recovered.json()["batchState"] == "unknown"
    assert recovered.json()["hasRunningBatch"] is False
    assert calls == []
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    batch = next(iter(persisted["versions"][f"{filename}:{fingerprint}"]["batches"].values()))
    assert batch["state"] == "unknown"
    assert "may have reached Gemini" in batch["message"]

    # Recovery never sends a replacement Gemini request by itself. A later
    # explicit Generate click uses a new request id and is allowed.
    response = _generate()
    assert response.status_code == 200
    assert calls == [True]


def test_unrelated_active_generation_does_not_hide_orphaned_batch(
    thumbnail_job, monkeypatch
):
    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, fingerprint = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    request_id = str(uuid.uuid4())
    state_path = thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    state_path.write_text(json.dumps({
        "versions": {
            f"{filename}:{fingerprint}": {
                "filename": filename,
                "sha256": fingerprint,
                "selected_candidate_id": None,
                "batches": {
                    request_id: {
                        "state": "running",
                        "requested_count": 1,
                        "started_at": "2026-10-18T00:00:00+00:00",
                        "candidates": [],
                    },
                },
            },
        },
    }), encoding="utf-8")
    app_module._clip_thumbnail_active_request = (
        "87654321-4321-4321-4321-cba987654321", 0, str(uuid.uuid4())
    )

    async def hold_unrelated_slot_and_read():
        await app_module._clip_thumbnail_generation_gate.acquire()
        try:
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.get(f"/api/clip-thumbnail/{JOB_ID}/0")
        finally:
            app_module._clip_thumbnail_generation_gate.release()
            app_module._clip_thumbnail_active_request = None

    response = asyncio.run(hold_unrelated_slot_and_read())

    assert response.status_code == 200
    assert response.json()["batchState"] == "unknown"
    assert response.json()["hasRunningBatch"] is False


def test_duplicate_request_id_recovers_its_orphan_before_returning(
    thumbnail_job, monkeypatch
):
    provider_calls = []
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: provider_calls.append(True) or _jpeg_bytes(),
    )
    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, fingerprint = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    request_id = str(uuid.uuid4())
    state_path = thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    state_path.write_text(json.dumps({
        "versions": {
            f"{filename}:{fingerprint}": {
                "filename": filename,
                "sha256": fingerprint,
                "selected_candidate_id": None,
                "batches": {
                    request_id: {
                        "state": "running",
                        "requested_count": 1,
                        "started_at": "2026-10-18T00:00:00+00:00",
                        "candidates": [],
                    },
                },
            },
        },
    }), encoding="utf-8")

    response = _generate(request_id=request_id)

    assert response.status_code == 409
    assert provider_calls == []
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    batch = persisted["versions"][f"{filename}:{fingerprint}"]["batches"][request_id]
    assert batch["state"] == "unknown"


def test_selection_rejects_a_candidate_whose_saved_hash_no_longer_matches(
    thumbnail_job, monkeypatch
):
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: _jpeg_bytes(),
    )
    response = _generate()
    assert response.status_code == 200
    candidate = response.json()["candidates"][0]

    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, _filename, fingerprint = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    root = thumbnail_job / "ai_thumbnails" / "0" / fingerprint
    (root / candidate["url"].rsplit("/", 1)[-1]).write_bytes(_different_jpeg_bytes())

    selected = _request(
        "post",
        f"/api/clip-thumbnail/{JOB_ID}/0/select",
        headers={"Origin": ORIGIN},
        json={"candidate_id": candidate["id"]},
    )

    assert selected.status_code == 409


def _seed_selected_ai_and_postiz_thumbnail(job_dir):
    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, fingerprint = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    candidate_id = "b" * 32
    content = _jpeg_bytes()
    root = app_module._clip_thumbnail_root(JOB_ID, 0, fingerprint)
    candidate_filename = f"candidate-{candidate_id}.jpg"
    (root / candidate_filename).write_bytes(content)
    app_module._write_clip_thumbnail_state(
        job_dir / app_module.CLIP_THUMBNAIL_STATE_FILE,
        {
            "versions": {
                f"{filename}:{fingerprint}": {
                    "filename": filename,
                    "sha256": fingerprint,
                    "selected_candidate_id": candidate_id,
                    "batches": {
                        str(uuid.uuid4()): {
                            "state": "completed",
                            "candidates": [{
                                "id": candidate_id,
                                "filename": candidate_filename,
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }],
                        },
                    },
                },
            },
        },
    )
    app_module._write_postiz_state(
        job_dir / app_module.POSTIZ_PUBLISH_STATE_FILE,
        {
            "versions": {
                f"{filename}:{fingerprint}": {
                    "state": "post_created",
                    "filename": filename,
                    "sha256": fingerprint,
                    "post_id": "post-1",
                    "action": "draft",
                    "media": {"id": "media-1", "path": "/uploads/clip.mp4"},
                    "thumbnail": {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "media": {"id": "thumbnail-1", "path": "/uploads/thumb.jpg"},
                    },
                },
            },
        },
    )
    return filename, fingerprint, candidate_id


def test_clear_thumbnail_choice_clears_ai_and_postiz_state_together(thumbnail_job):
    filename, fingerprint, _candidate_id = _seed_selected_ai_and_postiz_thumbnail(
        thumbnail_job
    )

    response = _request(
        "post",
        "/api/postiz/clear-thumbnail",
        headers={"Origin": ORIGIN},
        json={"job_id": JOB_ID, "clip_index": 0},
    )

    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert response.json()["selectedCandidateId"] is None
    assert response.json()["postizThumbnailSaved"] is False
    ai_state = app_module._read_clip_thumbnail_state(
        thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    )
    postiz_state = app_module._read_postiz_state(
        thumbnail_job / app_module.POSTIZ_PUBLISH_STATE_FILE
    )
    key = f"{filename}:{fingerprint}"
    assert ai_state["versions"][key]["selected_candidate_id"] is None
    assert "thumbnail" not in postiz_state["versions"][key]


def test_clear_thumbnail_choice_rolls_back_postiz_when_ai_write_fails(
    thumbnail_job, monkeypatch
):
    filename, fingerprint, candidate_id = _seed_selected_ai_and_postiz_thumbnail(
        thumbnail_job
    )
    original_write = app_module._write_clip_thumbnail_state

    def fail_ai_write(path, state):
        raise app_module.ClipThumbnailStateError(
            "simulated thumbnail state write failure"
        )

    monkeypatch.setattr(app_module, "_write_clip_thumbnail_state", fail_ai_write)
    response = _request(
        "post",
        "/api/postiz/clear-thumbnail",
        headers={"Origin": ORIGIN},
        json={"job_id": JOB_ID, "clip_index": 0},
    )
    monkeypatch.setattr(app_module, "_write_clip_thumbnail_state", original_write)

    assert response.status_code == 409
    ai_state = app_module._read_clip_thumbnail_state(
        thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    )
    postiz_state = app_module._read_postiz_state(
        thumbnail_job / app_module.POSTIZ_PUBLISH_STATE_FILE
    )
    key = f"{filename}:{fingerprint}"
    assert ai_state["versions"][key]["selected_candidate_id"] == candidate_id
    assert "thumbnail" in postiz_state["versions"][key]


def test_reference_failure_marks_reserved_batch_terminal(thumbnail_job, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_make_clip_reference_frame",
        lambda *_args: (_ for _ in ()).throw(
            app_module.ClipThumbnailStateError("reference frame failed")
        ),
    )

    response = _generate()

    assert response.status_code == 500
    state = app_module._read_clip_thumbnail_state(
        thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    )
    batch = next(iter(next(iter(state["versions"].values()))["batches"].values()))
    assert batch["state"] == "failed"


def test_reference_state_failure_before_gemini_is_definite_and_uses_no_quota(
    thumbnail_job, monkeypatch
):
    original_write = app_module._write_clip_thumbnail_state
    failed = {"value": False}
    provider_calls = []
    monkeypatch.setattr(
        app_module,
        "_make_clip_reference_frame",
        lambda *_args: (_jpeg_bytes(), "reference.jpg"),
    )
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: provider_calls.append(True) or _jpeg_bytes(),
    )

    def fail_reference_metadata_once(path, state):
        record = next(iter(state.get("versions", {}).values()), {})
        if not failed["value"] and "reference" in record:
            failed["value"] = True
            raise app_module.ClipThumbnailStateError(
                "reference metadata write failed"
            )
        return original_write(path, state)

    monkeypatch.setattr(
        app_module, "_write_clip_thumbnail_state", fail_reference_metadata_once
    )
    response = _generate()

    assert response.status_code == 409
    assert provider_calls == []
    state = app_module._read_clip_thumbnail_state(
        thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    )
    batch = next(iter(next(iter(state["versions"].values()))["batches"].values()))
    assert batch["state"] == "failed"
    assert "No image-generation request was made" in batch["message"]


def test_candidate_file_failure_keeps_prior_candidates_and_finishes_batch(
    thumbnail_job, monkeypatch
):
    writes = []
    original_write = app_module._write_thumbnail_file
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: _jpeg_bytes(),
    )

    def fail_second_candidate(path, content):
        writes.append(path)
        if len(writes) == 2:
            raise app_module.ClipThumbnailStateError("candidate disk write failed")
        return original_write(path, content)

    monkeypatch.setattr(app_module, "_write_thumbnail_file", fail_second_candidate)
    response = _generate(count=3)

    assert response.status_code == 200
    assert len(response.json()["candidates"]) == 1
    assert response.json()["batchState"] == "failed"
    assert "Generated 1 of 3 candidates" in response.json()["message"]


def test_final_state_write_failure_recovers_reserved_batch_to_terminal_unknown(
    thumbnail_job, monkeypatch
):
    original_write = app_module._write_clip_thumbnail_state
    fail_once = {"value": True}
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: _jpeg_bytes(),
    )

    def fail_completed_write(path, state):
        batches = next(iter(state.get("versions", {}).values()), {}).get("batches", {})
        if (
            fail_once["value"]
            and any(
                isinstance(batch, dict) and batch.get("state") == "completed"
                for batch in batches.values()
            )
        ):
            fail_once["value"] = False
            raise app_module.ClipThumbnailStateError("final state write failed")
        return original_write(path, state)

    monkeypatch.setattr(app_module, "_write_clip_thumbnail_state", fail_completed_write)
    response = _generate()

    assert response.status_code == 409
    state = app_module._read_clip_thumbnail_state(
        thumbnail_job / app_module.CLIP_THUMBNAIL_STATE_FILE
    )
    batch = next(iter(next(iter(state["versions"].values()))["batches"].values()))
    assert batch["state"] == "unknown"
