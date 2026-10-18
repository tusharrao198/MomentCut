import asyncio
import base64
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


def test_orphaned_running_batch_is_marked_unknown_before_a_new_request(
    thumbnail_job, monkeypatch
):
    monkeypatch.setattr(app_module, "_make_clip_reference_frame", lambda *_args: (None, None))
    monkeypatch.setattr(
        clip_thumbnail_image,
        "generate_clip_thumbnail",
        lambda *_args, **_kwargs: _jpeg_bytes(),
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

    response = _generate()

    assert response.status_code == 409
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    batch = next(iter(persisted["versions"][f"{filename}:{fingerprint}"]["batches"].values()))
    assert batch["state"] == "unknown"
    assert "may have reached Gemini" in batch["message"]


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
