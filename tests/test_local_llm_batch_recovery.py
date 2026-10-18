"""Regression coverage for malformed local transcript-selection responses."""

import json

import pytest

import gemini_worker

main = pytest.importorskip("main")


def _windows(count):
    return [
        {
            "id": f"window_{index:03d}",
            "start": float(index * 60),
            "end": float(index * 60 + 60),
            "text": f"Transcript window {index}",
        }
        for index in range(1, count + 1)
    ]


@pytest.fixture
def local_llm(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_local_malformed_batch_bisects_and_keeps_healthy_windows(
    monkeypatch, local_llm
):
    calls = []

    def fake_stage(client, model, prompt, schema):
        items = json.loads(prompt)
        calls.append([item["id"] for item in items])
        if len(items) > 1:
            raise gemini_worker.ModelResponseError(
                "Model response did not contain JSON."
            )
        item = items[0]
        return {"windows": [{"id": item["id"], "score": 50}]}, None

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    windows = _windows(4)
    result = main._run_stage_split(
        None, "qwen2.5:7b", windows, json.dumps, gemini_worker.ScoreResponse,
        "windows", [], "score")

    assert [entry["id"] for entry in result] == [window["id"] for window in windows]
    assert calls[0] == [window["id"] for window in windows]
    assert len(calls) == 7


def test_local_irrecoverable_single_window_is_skipped(monkeypatch, local_llm):
    calls = []

    def fake_stage(*args):
        calls.append(1)
        raise gemini_worker.ModelResponseError(
            "Model response did not contain JSON."
        )

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    result = main._run_stage_split(
        None, "qwen2.5:7b", _windows(1), json.dumps,
        gemini_worker.ScoreResponse, "windows", [], "score")

    assert result == []
    assert len(calls) == 1


def test_local_server_and_context_failures_remain_fatal(monkeypatch, local_llm):
    calls = []

    def fake_stage(*args):
        calls.append(1)
        raise RuntimeError(
            "LLM server 400 from http://llm.test/v1: context length exceeded"
        )

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    with pytest.raises(RuntimeError, match="context length exceeded"):
        main._run_stage_split(
            None, "qwen2.5:7b", _windows(4), json.dumps,
            gemini_worker.ScoreResponse, "windows", [], "score")
    assert len(calls) == 1
