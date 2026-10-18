"""Regression coverage for bounded local transcript recovery."""

import pytest

main = pytest.importorskip("main")


def test_local_score_batch_override_is_capped(monkeypatch, capsys):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LLM_SCORE_BATCH", "1000")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert main.score_batch_size() == 8
    assert "bounded local recovery; using 8" in capsys.readouterr().out
