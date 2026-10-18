"""The OpenAI-compatible moment-picker backend (llm_backend.py).

Self-hosters asked for a pipeline that never calls Google. The transcript
passes go to any /chat/completions server when LLM_BASE_URL is set; these
tests pin the contract main.py relies on: same (parsed, cost) shape as the
Gemini stage, schema validation, and the response_format fallback ladder.
"""
import json

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _serve(handler, monkeypatch):
    """Route llm_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_backend.active() is False
    assert llm_backend.describe() is None


def test_base_url_alone_activates_and_describes(local):
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1"}


def test_explicit_gemini_provider_wins_over_base_url(local, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.active() is False


# --- generate_json ---------------------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer ollama"  # documented placeholder when no key is set
    assert parsed["windows"][0]["score"] == 88
    assert cost["total_cost"] == 0.0 and cost["local"] is True
    assert cost["input_tokens"] == 120 and cost["output_tokens"] == 40


def test_falls_back_to_json_object_when_schema_mode_is_rejected(local, monkeypatch):
    formats = []

    def handler(request):
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({
            "windows": [{
                "id": "w0", "start": 0, "end": 90,
                "score": 50, "reason": "usable",
            }]
        })

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed["windows"][0]["id"] == "w0"


def test_generic_format_error_does_not_trigger_response_format_fallback(
    local, monkeypatch
):
    """A context/config error must fail once, rather than issue three calls."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            400, text="Input format exceeds this model's context window")

    _serve(handler, monkeypatch)
    with pytest.raises(RuntimeError, match="context window"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 1


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    def handler(request):
        text = "```json\n" + json.dumps({
            "windows": [{
                "id": "w0", "start": 0, "end": 90,
                "score": 50, "reason": "usable",
            }]
        }) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed["windows"][0]["id"] == "w0"


def test_salvages_valid_score_items_when_a_sibling_is_malformed(local, monkeypatch):
    _serve(lambda r: _completion({
        "windows": [
            {"id": "w0", "start": 0, "end": 90, "score": 85, "reason": "usable"},
            {"id": "broken"},
        ]
    }), monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed["windows"] == [{
        "id": "w0", "start": 0.0, "end": 90.0, "score": 85, "reason": "usable",
    }]


def test_salvages_a_local_score_without_nonessential_metadata(local, monkeypatch):
    """Scoring consumes only ID and score; canonical windows supply the rest."""
    _serve(lambda r: _completion({"windows": [{"id": "w0", "score": 85}]}),
           monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed["windows"] == [{
        "id": "w0", "start": 0.0, "end": 0.0, "score": 85, "reason": "",
    }]


def test_salvages_a_local_score_with_invalid_unused_metadata(local, monkeypatch):
    _serve(lambda r: _completion({
        "windows": [{
            "id": "w0", "start": "not-a-time", "end": None,
            "score": 85, "reason": None,
        }]
    }), monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed["windows"] == [{
        "id": "w0", "start": 0.0, "end": 0.0, "score": 85, "reason": "",
    }]


def test_salvages_valid_detail_items_when_a_sibling_is_malformed(local, monkeypatch):
    valid = {
        "start": 10, "end": 40, "source_window_id": "w0", "predicted_score": 85,
        "video_description_for_tiktok": "tiktok",
        "video_description_for_instagram": "instagram",
        "video_title_for_youtube_short": "title",
        "viral_hook_text": "hook",
    }
    _serve(lambda r: _completion({"shorts": [valid, {"source_window_id": "w1"}]}),
           monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.DetailResponse)
    assert len(parsed["shorts"]) == 1
    assert parsed["shorts"][0]["source_window_id"] == "w0"


def test_all_invalid_local_items_raise_to_trigger_split_or_singleton_skip(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(gemini_worker.ModelResponseError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "no valid 'windows' items" in str(exc.value)


def test_empty_local_items_raise_to_trigger_split_or_singleton_skip(local, monkeypatch):
    _serve(lambda r: _completion({"windows": []}), monkeypatch)
    with pytest.raises(gemini_worker.ModelResponseError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "did not contain any 'windows' items" in str(exc.value)


def test_all_invalid_local_detail_items_raise_to_trigger_split_or_singleton_skip(
    local, monkeypatch
):
    _serve(lambda r: _completion({"shorts": [{"source_window_id": "w0"}]}),
           monkeypatch)
    with pytest.raises(gemini_worker.ModelResponseError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.DetailResponse)
    assert "no valid 'shorts' items" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


# --- main.py routing (needs the heavy deps; skipped on minimal CI) ------------

def test_stage_routes_to_local_backend_and_retries_transient(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # Actual httpx messages vary; this one contains none of the
            # old string tokens, so retry must come from exception typing.
            raise httpx.ConnectError("offline")
        return {"windows": [{"id": "w0", "start": 0, "end": 1, "score": 50, "reason": ""}]}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_gemini_stage(None, "qwen2.5:14b", "prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 2
    assert parsed["windows"][0]["score"] == 50
    assert cost["total_cost"] == 0.0


def test_score_batch_shrinks_for_local_models(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 3
    monkeypatch.setenv("LLM_SCORE_BATCH", "5")
    assert main.score_batch_size() == 5
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_SCORE_BATCH")
    assert main.score_batch_size() == 8
