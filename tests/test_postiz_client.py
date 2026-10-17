import json

import httpx
import pytest

import postiz_client as pc


@pytest.fixture(autouse=True)
def postiz_env(monkeypatch):
    for name in (
        "POSTIZ_API_BASE_URL",
        "POSTIZ_API_KEY",
        "POSTIZ_YOUTUBE_INTEGRATION_ID",
        "POSTIZ_DEFAULT_POST_TYPE",
        "POSTIZ_YOUTUBE_PRIVACY_STATUS",
        "POSTIZ_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POSTIZ_API_BASE_URL", pc.EXPECTED_API_BASE_URL)
    monkeypatch.setenv("POSTIZ_API_KEY", "postiz-test-key")


def _install_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        pc,
        "_client",
        lambda _config: httpx.Client(transport=transport, follow_redirects=False),
    )


def test_configuration_requires_exact_private_docker_url(monkeypatch):
    for bad in (
        "http://postiz:5001/api/public/v1",
        "https://postiz:5000/api/public/v1",
        "http://user@postiz:5000/api/public/v1",
        "http://postiz:5000/api/public/v1?send=elsewhere",
    ):
        monkeypatch.setenv("POSTIZ_API_BASE_URL", bad)
        with pytest.raises(pc.PostizError, match="private Docker network"):
            pc.configuration()


@pytest.mark.parametrize("name,value", [
    ("POSTIZ_DEFAULT_POST_TYPE", "schedule"),
    ("POSTIZ_YOUTUBE_PRIVACY_STATUS", "friends"),
    ("POSTIZ_TIMEOUT_SECONDS", "12"),
    ("POSTIZ_TIMEOUT_SECONDS", "not-a-number"),
])
def test_configuration_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(pc.PostizError):
        pc.configuration()


def test_upload_uses_raw_authorization_header_and_streams_file(tmp_path, monkeypatch):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(201, json={"id": "media-1", "path": "/uploads/clip.mp4"})

    _install_transport(monkeypatch, handler)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"small-video")

    media = pc.upload_file(pc.configuration(), clip, "clip.mp4")

    assert media == {"id": "media-1", "path": "/uploads/clip.mp4"}
    assert seen["method"] == "POST"
    assert seen["url"] == f"{pc.EXPECTED_API_BASE_URL}/upload"
    assert seen["authorization"] == "postiz-test-key"
    assert b'name="file"' in seen["body"]
    assert b"small-video" in seen["body"]


def test_upload_rejects_redirect_and_malformed_response(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    _install_transport(
        monkeypatch,
        lambda _request: httpx.Response(302, headers={"Location": "http://elsewhere/"}),
    )
    with pytest.raises(pc.PostizError, match="media upload"):
        pc.upload_file(pc.configuration(), clip, "clip.mp4")

    _install_transport(monkeypatch, lambda _request: httpx.Response(201, json={"path": "/x"}))
    with pytest.raises(pc.PostizError, match="usable media"):
        pc.upload_file(pc.configuration(), clip, "clip.mp4")


def test_upload_thumbnail_uses_raw_authorization_header_and_validates_input(monkeypatch):
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(
            201, json={"id": "thumbnail-1", "path": "/uploads/thumbnail.png"}
        )

    _install_transport(monkeypatch, handler)
    thumbnail = b"\x89PNG\r\n\x1a\nsmall-thumbnail"
    media = pc.upload_thumbnail(
        pc.configuration(), thumbnail, "thumbnail.png", "image/png"
    )

    assert media == {"id": "thumbnail-1", "path": "/uploads/thumbnail.png"}
    assert seen["authorization"] == "postiz-test-key"
    assert b'name="file"' in seen["body"]
    assert thumbnail in seen["body"]

    with pytest.raises(pc.PostizError, match="JPEG or PNG"):
        pc.upload_thumbnail(pc.configuration(), b"image", "thumbnail.gif", "image/gif")
    with pytest.raises(pc.PostizError, match="2 MB"):
        pc.upload_thumbnail(
            pc.configuration(),
            b"x" * (pc.MAX_THUMBNAIL_BYTES + 1),
            "thumbnail.png",
            "image/png",
        )


def test_resolve_youtube_requires_enabled_identifier_and_unambiguous_choice(monkeypatch):
    config = pc.configuration()

    def handler(_request):
        return httpx.Response(200, json=[
            {"id": "disabled", "identifier": "youtube", "disabled": True},
            {"id": "other", "identifier": "tiktok", "disabled": False},
            {"id": "", "identifier": "youtube", "disabled": False},
            {"id": "yt-1", "identifier": "youtube", "disabled": False, "name": "Channel"},
        ])

    _install_transport(monkeypatch, handler)
    assert pc.resolve_youtube_integration(config)["id"] == "yt-1"

    monkeypatch.setenv("POSTIZ_YOUTUBE_INTEGRATION_ID", "disabled")
    with pytest.raises(pc.PostizError, match="not an enabled YouTube"):
        pc.resolve_youtube_integration(pc.configuration())

    def multiple(_request):
        return httpx.Response(200, json=[
            {"id": "yt-1", "identifier": "youtube", "disabled": False},
            {"id": "yt-2", "identifier": "youtube", "disabled": False},
        ])

    _install_transport(monkeypatch, multiple)
    monkeypatch.delenv("POSTIZ_YOUTUBE_INTEGRATION_ID", raising=False)
    with pytest.raises(pc.PostizError, match="More than one"):
        pc.resolve_youtube_integration(pc.configuration())


@pytest.mark.parametrize("media", [
    {},
    {"id": "", "path": "/uploads/clip.mp4"},
    {"id": "media-1", "path": ""},
])
def test_create_post_rejects_invalid_saved_media_without_network_call(monkeypatch, media):
    monkeypatch.setattr(
        pc,
        "_client",
        lambda *_args, **_kwargs: pytest.fail("must not call Postiz"),
    )
    with pytest.raises(pc.PostizError, match="media state"):
        pc.create_youtube_post(pc.configuration(), "yt-1", media, "Valid title", "")


def test_create_post_has_exact_youtube_media_shape_and_parses_array_response(monkeypatch):
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=[{"postId": "post-123"}])

    _install_transport(monkeypatch, handler)
    monkeypatch.setattr(pc, "_iso_now", lambda: "2026-10-17T00:00:00.000Z")
    post_id = pc.create_youtube_post(
        pc.configuration(),
        "yt-1",
        {"id": "media-1", "path": "/uploads/clip.mp4"},
        "A valid title",
        "Description",
    )

    assert post_id == "post-123"
    assert seen["authorization"] == "postiz-test-key"
    assert seen["body"] == {
        "type": "draft",
        "date": "2026-10-17T00:00:00.000Z",
        "shortLink": False,
        "tags": [],
        "posts": [{
            "integration": {"id": "yt-1"},
            "value": [{"content": "Description", "image": [{
                "id": "media-1", "path": "/uploads/clip.mp4",
            }]}],
            "settings": {
                "__type": "youtube",
                "title": "A valid title",
                "type": "private",
                "selfDeclaredMadeForKids": "no",
                "tags": [],
            },
        }],
    }


def test_create_post_includes_optional_youtube_thumbnail(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=[{"postId": "post-123"}])

    _install_transport(monkeypatch, handler)
    post_id = pc.create_youtube_post(
        pc.configuration(),
        "yt-1",
        {"id": "media-1", "path": "/uploads/clip.mp4"},
        "A valid title",
        "Description",
        {"id": "thumbnail-1", "path": "/uploads/thumbnail.png"},
    )

    assert post_id == "post-123"
    assert seen["body"]["posts"][0]["settings"]["thumbnail"] == {
        "id": "thumbnail-1",
        "path": "/uploads/thumbnail.png",
    }


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="temporary"),
    httpx.Response(201, json=[]),
    httpx.Response(201, json=[{}]),
])
def test_create_post_marks_ambiguous_responses_unknown(monkeypatch, response):
    _install_transport(monkeypatch, lambda _request: response)
    with pytest.raises(pc.PostizOutcomeUnknown):
        pc.create_youtube_post(
            pc.configuration(), "yt-1", {"id": "media", "path": "/upload"},
            "Valid title", "",
        )


def test_create_post_treats_definite_client_error_as_retryable(monkeypatch):
    _install_transport(monkeypatch, lambda _request: httpx.Response(400, json={"message": "bad title"}))
    with pytest.raises(pc.PostizError, match="bad title") as exc:
        pc.create_youtube_post(
            pc.configuration(), "yt-1", {"id": "media", "path": "/upload"},
            "Valid title", "",
        )
    assert not isinstance(exc.value, pc.PostizOutcomeUnknown)
