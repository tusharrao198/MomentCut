import asyncio
import json
import os

import httpx
import pytest

import postiz_client

app_module = pytest.importorskip("app")

JOB_ID = "12345678-1234-1234-1234-123456789abc"
ORIGIN = "http://localhost:5175"


def _request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await getattr(client, method)(path, **kwargs)
    return asyncio.run(send())


@pytest.fixture()
def postiz_job(tmp_path, monkeypatch):
    output = tmp_path / "output"
    job_dir = output / JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "source_metadata.json").write_text("{}", encoding="utf-8")
    (job_dir / "clip.mp4").write_bytes(b"rendered-video")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(output))
    monkeypatch.setattr(app_module, "jobs", {
        JOB_ID: {
            "result": {
                "clips": [{
                    "video_url": f"/videos/{JOB_ID}/clip.mp4",
                    "video_title_for_youtube_short": "A valid YouTube title",
                    "video_description_for_tiktok": "A description",
                }],
            },
        },
    })
    monkeypatch.setattr(app_module, "_postiz_publish_locks", {})
    config = postiz_client.PostizConfig(
        api_base_url=postiz_client.EXPECTED_API_BASE_URL,
        api_key="not-exposed",
        youtube_integration_id="yt-1",
        post_type="draft",
        privacy_status="private",
        timeout_seconds=300,
    )
    monkeypatch.setattr(postiz_client, "configuration", lambda: config)
    return job_dir, config


def _publish():
    return _request(
        "post",
        "/api/postiz/publish",
        headers={"Origin": ORIGIN},
        json={"job_id": JOB_ID, "clip_index": 0},
    )


def test_postiz_publish_rejects_hostile_browser_origin():
    response = _request(
        "post",
        "/api/postiz/publish",
        headers={"Origin": "https://attacker.example"},
        json={"job_id": JOB_ID, "clip_index": 0},
    )
    assert response.status_code == 403
    assert "local OpenShorts dashboard" in response.json()["detail"]


def test_self_host_cors_rejects_hostile_preflight():
    response = _request(
        "options",
        "/api/postiz/publish",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") is None


def test_public_config_never_returns_postiz_credentials(monkeypatch):
    monkeypatch.setenv("POSTIZ_API_BASE_URL", postiz_client.EXPECTED_API_BASE_URL)
    monkeypatch.setenv("POSTIZ_API_KEY", "must-not-leak")
    response = _request("get", "/api/config")
    payload = response.json()

    assert payload["postizConfigured"] is True
    assert payload["postizPostType"] == "draft"
    assert "POSTIZ_API_KEY" not in payload
    assert "must-not-leak" not in json.dumps(payload)


def test_postiz_publish_creates_once_then_reuses_same_rendered_version(postiz_job, monkeypatch):
    uploaded, created = [], []
    monkeypatch.setattr(postiz_client, "resolve_youtube_integration",
                        lambda _config: {"id": "yt-1", "identifier": "youtube"})
    monkeypatch.setattr(
        postiz_client, "upload_file",
        lambda _config, path, filename: uploaded.append((str(path), filename))
        or {"id": "media-1", "path": "/uploads/clip.mp4"},
    )
    monkeypatch.setattr(
        postiz_client, "create_youtube_post",
        lambda *_args: created.append(True) or "post-1",
    )

    first = _publish()
    second = _publish()

    assert first.status_code == 200
    assert first.json()["reused"] is False
    assert second.status_code == 200
    assert second.json()["reused"] is True
    assert uploaded == [(str(postiz_job[0] / "clip.mp4"), "clip.mp4")]
    assert created == [True]


def test_postiz_pending_state_fails_closed_after_restart(postiz_job, monkeypatch):
    clip = app_module.jobs[JOB_ID]["result"]["clips"][0]
    _, filename, digest = app_module._resolve_postiz_clip(JOB_ID, [clip], 0)
    state_path = app_module._postiz_state_path(JOB_ID)
    app_module._write_postiz_state(state_path, {
        "versions": {
            f"{filename}:{digest}": {
                "state": "post_create_pending",
                "filename": filename,
                "sha256": digest,
                "media": {"id": "media-1", "path": "/uploads/clip.mp4"},
            },
        },
    })
    monkeypatch.setattr(
        postiz_client,
        "create_youtube_post",
        lambda *_args: pytest.fail("must not create a second post"),
    )

    response = _publish()

    assert response.status_code == 409
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["versions"][f"{filename}:{digest}"]["state"] == "post_create_unknown"


def test_postiz_rate_limit_is_preserved_for_retry(postiz_job, monkeypatch):
    monkeypatch.setattr(postiz_client, "resolve_youtube_integration",
                        lambda _config: {"id": "yt-1", "identifier": "youtube"})
    monkeypatch.setattr(
        postiz_client,
        "upload_file",
        lambda *_args: {"id": "media-1", "path": "/uploads/clip.mp4"},
    )
    monkeypatch.setattr(
        postiz_client,
        "create_youtube_post",
        lambda *_args: (_ for _ in ()).throw(
            postiz_client.PostizError("Postiz post creation failed (429): slow down", 429)
        ),
    )

    response = _publish()

    assert response.status_code == 429
    assert "slow down" in response.json()["detail"]


def test_postiz_handoff_keeps_file_and_metadata_from_one_clip_snapshot(postiz_job, monkeypatch):
    captured = {}
    original_resolve = app_module._resolve_postiz_clip

    def resolve_then_change_live_clip(*args):
        result = original_resolve(*args)
        app_module.jobs[JOB_ID]["result"]["clips"][0].update({
            "video_url": f"/videos/{JOB_ID}/different.mp4",
            "video_title_for_youtube_short": "Wrong newer title",
            "video_description_for_tiktok": "Wrong newer description",
        })
        return result

    monkeypatch.setattr(app_module, "_resolve_postiz_clip", resolve_then_change_live_clip)
    monkeypatch.setattr(postiz_client, "resolve_youtube_integration",
                        lambda _config: {"id": "yt-1", "identifier": "youtube"})
    monkeypatch.setattr(
        postiz_client,
        "upload_file",
        lambda *_args: {"id": "media-1", "path": "/uploads/clip.mp4"},
    )
    monkeypatch.setattr(
        postiz_client,
        "create_youtube_post",
        lambda _config, _integration_id, _media, title, description:
        captured.update(title=title, description=description) or "post-1",
    )

    response = _publish()

    assert response.status_code == 200
    assert captured == {
        "title": "A valid YouTube title",
        "description": "A description",
    }


@pytest.mark.parametrize("index", [-1, 1])
def test_postiz_publish_rejects_negative_and_out_of_range_indexes(postiz_job, index):
    response = _request(
        "post",
        "/api/postiz/publish",
        headers={"Origin": ORIGIN},
        json={"job_id": JOB_ID, "clip_index": index},
    )
    assert response.status_code == 404


def test_postiz_clip_rejects_symlink_outside_job(postiz_job):
    job_dir, _config = postiz_job
    outside = job_dir.parent / "secret.mp4"
    outside.write_bytes(b"not-a-clip")
    link = job_dir / "linked.mp4"
    try:
        os.symlink(outside, link)
    except (AttributeError, NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this filesystem")
    clip = {"video_url": f"/videos/{JOB_ID}/linked.mp4"}
    with pytest.raises(app_module.PostizStateError, match="outside"):
        app_module._resolve_postiz_clip(JOB_ID, [clip], 0)


def test_postiz_publish_changed_rendered_file_is_a_new_version(postiz_job, monkeypatch):
    job_dir, _config = postiz_job
    uploaded, created = [], []
    monkeypatch.setattr(postiz_client, "resolve_youtube_integration",
                        lambda _config: {"id": "yt-1", "identifier": "youtube"})
    monkeypatch.setattr(
        postiz_client, "upload_file",
        lambda _config, path, filename: uploaded.append(filename)
        or {"id": f"media-{len(uploaded)}", "path": f"/uploads/{filename}"},
    )
    monkeypatch.setattr(
        postiz_client, "create_youtube_post",
        lambda *_args: created.append(True) or f"post-{len(created)}",
    )

    assert _publish().status_code == 200
    (job_dir / "subtitled_clip.mp4").write_bytes(b"new-rendered-video")
    app_module.jobs[JOB_ID]["result"]["clips"][0]["video_url"] = (
        f"/videos/{JOB_ID}/subtitled_clip.mp4"
    )
    assert _publish().status_code == 200

    assert uploaded == ["clip.mp4", "subtitled_clip.mp4"]
    assert created == [True, True]
