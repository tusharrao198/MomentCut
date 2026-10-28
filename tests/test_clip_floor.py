"""The clip floor must survive a model that ignores it.

`min_clips` lived only in the detail prompt, so a run that came back with
half the requested clips shipped half the clips: the measured 9:10 source
asked for 6-12 and delivered 3-4. The windows the detail pass skipped get
one more call before the job settles for less.
"""
import json
import re

import pytest

import audio_events

main = pytest.importorskip("main")


def _transcript(n_segments=40):
    segments = []
    for i in range(n_segments):
        start = i * 15.0
        segments.append({
            "start": start, "end": start + 15.0,
            "text": f"sentence number {i} with enough words to score",
            "words": [{"word": f"w{i}", "start": start, "end": start + 0.4}],
        })
    return {"language": "es", "segments": segments}


def _clip(start, window_id, score=80):
    return {
        "start": start, "end": start + 30.0, "source_window_id": window_id,
        "predicted_score": score, "video_description_for_tiktok": "d",
        "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "t", "viral_hook_text": "h",
        "why": "because",
    }


def _run(monkeypatch, detail_responses, stage_cost=None, return_result=False,
         audio_event_artifact=None, use_local=False):
    """Drive get_viral_clips with canned stage results; returns (clips, calls)."""
    calls = []

    def fake_stage(client, model, prompt, schema):
        if schema is main.gemini_worker.LocalWindowDetailResponse:
            match = re.search(r"^WINDOW_ID: ([A-Za-z0-9_]+)$", prompt, re.M)
            window_id = match.group(1)
            calls.append(("detail-window", [window_id], prompt))
            available = detail_responses[0] if detail_responses else []
            clip = next(
                (
                    item for item in available
                    if item.get("source_window_id") == window_id
                ),
                None,
            )
            return {"clip": clip}, stage_cost
        if schema is main.gemini_worker.LocalClipRankResponse:
            payload = json.loads(
                prompt.split("CANDIDATES_JSON:")[-1].split("Return only:")[0]
            )
            calls.append((
                "global-rank",
                [item["candidate_id"] for item in payload],
            ))
            return {
                "candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "score": item.get("preliminary_score") or 0,
                        "reason": "same order",
                    }
                    for item in payload
                ]
            }, stage_cost
        payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0]
                             if "WINDOWS_JSON:" in prompt else "[]")
        if schema is main.gemini_worker.ScoreResponse:
            ids = [w["id"] for w in payload]
            calls.append(("score", ids, payload))
            # The prompt asks for every window to be scored.
            return {"windows": [{"id": i, "start": 0, "end": 90, "score": 90 - n,
                                 "reason": "r"}
                                for n, i in enumerate(ids)]}, stage_cost
        windows = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1]
                             .split("Return only:")[0])
        ids = [w["id"] for w in windows]
        calls.append(("detail", ids))
        return {"shorts": detail_responses.pop(0)}, stage_cost

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    if use_local:
        # The local aggregation test deliberately exercises the OpenAI-style
        # route with a canned stage response.
        monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
        monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    else:
        # The Gemini-stage fixtures must not inherit a developer's local
        # Ollama configuration or disabled Gemini feature flag.
        for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_PROVIDER"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("GEMINI_CLIP_ANALYSIS_ENABLED", "true")
        # Score every fixture window together so the canned order maps to the
        # candidate ownership asserted below.
        monkeypatch.setenv("LLM_SCORE_BATCH", "10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda api_key=None: object())
    out = main.get_viral_clips(
        _transcript(), 600.0, audio_event_artifact=audio_event_artifact,
    )
    return (out if return_result else (out or {}).get("shorts", [])), calls


def _audio_event_artifact():
    return {
        "schema_version": 1,
        "provider": "ced",
        "profile": {
            "id": "gameplay-v1", "version": 1,
            "sha256": audio_events._profile_digest(),
        },
        "identity": {
            "source_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "worker_build": "test", "sample_rate_hz": 16000,
        },
        "runtime": {"requested_device": "metal", "observed_device": "MTL0"},
        "audio_stream": {
            "absolute_stream_index": 0, "start_seconds": 0.0,
            "selection_rule": "default",
        },
        "events": [{
            "label": "Explosion", "start_seconds": 120.0,
            "end_seconds": 130.0, "max_score": 0.9,
        }],
        "summary": {
            "qualified_event_count": 1, "window_count": 1,
            "gpu_residency_note": "test",
        },
    }


@pytest.mark.parametrize("use_local", [False, True])
def test_ced_evidence_reaches_each_selector_payload(monkeypatch, use_local):
    enough = [_clip(i * 60, f"window_{i + 1:03d}") for i in range(6)]
    _clips, calls = _run(
        monkeypatch,
        [enough],
        audio_event_artifact=_audio_event_artifact(),
        use_local=use_local,
    )
    score_payloads = [entry[2] for entry in calls if entry[0] == "score"]
    assert any(
        item.get("audio_events")
        for payload in score_payloads
        for item in payload
    )
    if use_local:
        assert any(
            "AUDIO_EVENT_EVIDENCE" in entry[2]
            for entry in calls if entry[0] == "detail-window"
        )


def test_a_short_detail_result_asks_the_unused_windows(monkeypatch):
    # Two clips back on a shortlist that asked for six: the four windows that
    # produced nothing get a second call, and its clips are kept.
    first = [_clip(0, "window_001"), _clip(120, "window_003")]
    second = [_clip(240, "window_005"), _clip(300, "window_006"),
              _clip(360, "window_007"), _clip(420, "window_008")]
    clips, calls = _run(monkeypatch, [first, second])

    detail_calls = [c for c in calls if c[0] == "detail"]
    assert len(detail_calls) == 2, "the floor never triggered a second call"
    # The retry only revisits windows the first pass returned nothing for.
    assert "window_001" not in detail_calls[1][1]
    assert "window_003" not in detail_calls[1][1]
    assert len(clips) >= 6
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)


def test_no_second_call_when_the_floor_is_met(monkeypatch):
    enough = [_clip(i * 60, f"window_{i + 1:03d}") for i in range(6)]
    clips, calls = _run(monkeypatch, [enough])
    assert len([c for c in calls if c[0] == "detail"]) == 1
    assert len(clips) == 6


def test_an_empty_retry_is_accepted_rather_than_padded(monkeypatch):
    # Material that genuinely holds two clips still returns two, not filler.
    clips, calls = _run(monkeypatch, [[_clip(0, "window_001")], []])
    assert len([c for c in calls if c[0] == "detail"]) == 2
    assert len(clips) == 1


def test_scoring_fills_the_shortlist_it_targets(monkeypatch):
    # 600s -> target 8. Every window is scored, so the shortlist is the top 8
    # globally instead of the 3-per-batch a constant cap allowed.
    clips, calls = _run(monkeypatch, [[_clip(i * 60, f"window_{i + 1:03d}")
                                       for i in range(6)]])
    detail_windows = [c[1] for c in calls if c[0] == "detail"][0]
    assert len(detail_windows) == 8


def test_local_cost_metadata_survives_two_pass_aggregation(monkeypatch):
    """The dashboard needs this flag to avoid displaying local work as Gemini."""
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    local_cost = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_cost": 0.0,
        "local": True,
    }
    enough = [_clip(i * 60, f"window_{i + 1:03d}") for i in range(6)]
    result, _ = _run(
        monkeypatch,
        [enough],
        stage_cost=local_cost,
        return_result=True,
        use_local=True,
    )
    assert result["cost_analysis"]["local"] is True
    assert result["cost_analysis"]["model"] == "qwen2.5:7b"
