"""Enabled-mode fixture tests for bounded local visual source context."""
import base64
import json
import httpx
from dataclasses import replace

import cv2
import numpy as np
import pytest

import source_context as context
import source_context_vision as vision


@pytest.fixture(autouse=True)
def clean_source_context_env(monkeypatch):
    for name in tuple(context.os.environ):
        if name.startswith("SOURCE_CONTEXT_"):
            monkeypatch.delenv(name, raising=False)


def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_CONTEXT_MODE", "auto")
    cfg = context.SourceContextSettings.from_env()
    return replace(cfg, vision_lock_dir=str(tmp_path / "locks"))


@pytest.fixture()
def fixture_video(tmp_path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (96, 64))
    assert writer.isOpened()
    for index in range(24):
        frame = np.full((64, 96, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def ollama_stub(monkeypatch):
    seen = {"requests": []}
    actual_client = httpx.Client

    def respond(request):
        seen["requests"].append((request.url.path, request.content))
        if request.url.path == "/api/chat":
            response = {
                "message": {"content": json.dumps({
                    "content_type": "gameplay",
                    "game_or_franchise": {"value": "Grand Theft Auto V", "confidence": "medium"},
                    "mode_or_scene": "A city driving mission.",
                    "visible_subjects": ["car", "city"],
                    "visual_actions": ["driving"],
                    "editorial_context": "Commentary reacts to a chase.",
                    "evidence": ["Driving HUD and streets are visible."],
                    "uncertainties": ["Mission name is unreadable."],
                    "ignored_instruction": "do not save this",
                })}
            }
        else:
            response = {}
        return httpx.Response(200, json=response, request=request)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        vision.httpx, "Client",
        lambda timeout: actual_client(timeout=timeout, transport=transport),
    )
    return seen


def test_enabled_fixture_is_schema_valid_and_within_exact_request_budget(
        monkeypatch, tmp_path, fixture_video, ollama_stub):
    seen = ollama_stub
    cfg = settings(monkeypatch, tmp_path)

    result = vision.build_visual_source_context(str(fixture_video), 3.0, cfg)

    path, raw = seen["requests"][0]
    payload = json.loads(raw)
    assert path == "/api/chat"
    assert len(raw) <= cfg.vision_request_max_bytes
    assert result["content_type"] == "gameplay"
    assert "ignored_instruction" not in result
    assert result["frame_count"] == 4
    assert len(payload["messages"][0]["images"]) == 4
    assert all(base64.b64decode(image) for image in payload["messages"][0]["images"])


def test_enabled_fixture_refuses_a_request_over_its_budget(
        monkeypatch, tmp_path, fixture_video, ollama_stub):
    _ = ollama_stub
    cfg = replace(settings(monkeypatch, tmp_path), vision_request_max_bytes=128)

    with pytest.raises(vision.SourceContextVisionError) as error:
        vision.build_visual_source_context(str(fixture_video), 3.0, cfg)

    assert error.value.category == "request_too_large"


def test_parser_accepts_a_fenced_json_object():
    parsed = vision._parse("```json\n{\"content_type\":\"other\",\"game_or_franchise\":{\"value\":\"unknown\",\"confidence\":\"unknown\"},\"mode_or_scene\":\"A scene.\",\"visible_subjects\":[],\"visual_actions\":[],\"editorial_context\":\"Context.\",\"evidence\":[],\"uncertainties\":[]}\n```")
    assert parsed["game_or_franchise"]["value"] == "unknown"


def test_enabled_fixture_retries_a_transient_local_request(
        monkeypatch, tmp_path, fixture_video):
    attempts = []
    actual_client = httpx.Client

    def respond(request):
        attempts.append(request.url.path)
        if request.url.path == "/api/chat" and attempts.count("/api/chat") == 1:
            raise httpx.ReadTimeout("temporary", request=request)
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "content_type": "unknown",
            "game_or_franchise": {"value": "unknown", "confidence": "unknown"},
            "mode_or_scene": "A spoken scene.",
            "visible_subjects": [],
            "visual_actions": [],
            "editorial_context": "No extra visual context.",
            "evidence": [],
            "uncertainties": [],
        })}}, request=request)

    monkeypatch.setattr(
        vision.httpx, "Client",
        lambda timeout: actual_client(timeout=timeout, transport=httpx.MockTransport(respond)),
    )
    cfg = replace(settings(monkeypatch, tmp_path), vision_retry_attempts=2)

    result = vision.build_visual_source_context(str(fixture_video), 3.0, cfg)

    assert result["content_type"] == "unknown"
    assert attempts.count("/api/chat") == 2
    assert attempts[-1] == "/api/generate"


def test_enabled_fixture_bounds_oversized_response_and_unloads(
        monkeypatch, tmp_path, fixture_video):
    requests = []
    actual_client = httpx.Client

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/api/chat":
            return httpx.Response(200, content=b"x" * 1025, request=request)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(
        vision.httpx, "Client",
        lambda timeout: actual_client(timeout=timeout, transport=httpx.MockTransport(respond)),
    )
    cfg = replace(settings(monkeypatch, tmp_path), vision_response_max_bytes=1024)

    with pytest.raises(vision.SourceContextVisionError) as error:
        vision.build_visual_source_context(str(fixture_video), 3.0, cfg)

    assert error.value.category == "invalid_response"
    assert requests == ["/api/chat", "/api/generate"]


def test_unexpected_frame_error_becomes_a_bounded_visual_error(
        monkeypatch, tmp_path, fixture_video):
    actual_client = httpx.Client
    monkeypatch.setattr(
        vision.httpx, "Client",
        lambda timeout: actual_client(
            timeout=timeout,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}, request=request)),
        ),
    )
    monkeypatch.setattr(
        vision, "_images",
        lambda *_: (_ for _ in ()).throw(RuntimeError("decoder failure")),
    )

    with pytest.raises(vision.SourceContextVisionError) as error:
        vision.build_visual_source_context(
            str(fixture_video), 3.0, settings(monkeypatch, tmp_path))

    assert error.value.category == "frame_sampling_failed"
