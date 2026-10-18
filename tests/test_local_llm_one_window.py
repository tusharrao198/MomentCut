"""Regression coverage for local transcript selection, one window at a time.

Qwen is materially more reliable when a detail request contains one transcript
window and asks for at most one clip.  These tests pin that local-only contract:
validate immediately, retry a semantically invalid boundary once, expand into
lower-ranked windows only when the clip floor is still short, then rank the
surviving candidates globally without ever losing one to malformed rank data.
"""

import re

import pytest

from clip_selection import (
    apply_global_candidate_ranking,
    dedupe_ranked_with_floor,
)
import gemini_worker

main = pytest.importorskip("main")


@pytest.fixture(autouse=True)
def _local_qwen(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_DETAIL_FALLBACK_WINDOWS", raising=False)


def _window(number, score=80):
    start = (number - 1) * 100.0
    return {
        "id": f"window_{number:03d}",
        "start": start,
        "end": start + 90.0,
        "text": f"transcript window {number}",
        "score": score,
    }


def _clip(window, *, start_offset=10.0, score=80):
    start = float(window["start"]) + start_offset
    return {
        "start": start,
        "end": start + 30.0,
        # Deliberately wrong: the application must replace this with the one
        # canonical window that was actually sent to the model.
        "source_window_id": "model-invented-id",
        "predicted_score": score,
        "video_description_for_tiktok": "TikTok copy",
        "video_description_for_instagram": "Instagram copy",
        "video_title_for_youtube_short": f"Moment from {window['id']}",
        "viral_hook_text": "Watch this",
        "why": "Clear hook and payoff.",
        "youtube_description_intro": "A grounded description of this moment.",
        "youtube_tags": ["specific tag"],
        "youtube_hashtags": ["#shorts"],
        "thumbnail_brief": {},
    }


def _detail_call(
    monkeypatch,
    primary,
    fallback,
    *,
    min_clips=2,
    max_clips=4,
):
    return main._run_local_detail_windows(
        client=None,
        model_name="qwen2.5:7b",
        primary_windows=primary,
        fallback_windows=fallback,
        video_duration=1000.0,
        language="en",
        min_secs=15.0,
        max_secs=60.0,
        min_clips=min_clips,
        max_clips=max_clips,
        costs=[],
    )


def _ids_in_prompt(prompt):
    return re.findall(r"window_\d{3}", prompt)


def test_local_detail_sends_exactly_one_primary_window_per_call(monkeypatch):
    primary = [_window(1), _window(2), _window(3)]
    prompts = []

    def fake_stage(_client, _model, prompt, schema):
        assert schema is gemini_worker.LocalWindowDetailResponse
        prompts.append(prompt)
        window_id = _ids_in_prompt(prompt)
        assert len(set(window_id)) == 1
        window = next(item for item in primary if item["id"] == window_id[0])
        return {"clip": _clip(window)}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    clips = _detail_call(
        monkeypatch, primary, [], min_clips=2
    )

    assert len(prompts) == len(primary)
    assert [set(_ids_in_prompt(prompt)).pop() for prompt in prompts] == [
        item["id"] for item in primary
    ]
    assert [clip["source_window_id"] for clip in clips] == [
        item["id"] for item in primary
    ]


def test_semantically_invalid_clip_retries_once_for_that_window(monkeypatch):
    window = _window(1)
    calls = []

    def fake_stage(_client, _model, prompt, schema):
        assert schema is gemini_worker.LocalWindowDetailResponse
        calls.append(prompt)
        candidate = _clip(window)
        if len(calls) == 1:
            candidate["start"] = window["end"] + 10
            candidate["end"] = window["end"] + 40
        return {"clip": candidate}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    clips = _detail_call(
        monkeypatch, [window], [], min_clips=1
    )

    assert len(calls) == 2
    assert len(clips) == 1
    assert clips[0]["source_window_id"] == window["id"]
    assert "previous" in calls[1].lower() or "invalid" in calls[1].lower()


def test_null_clip_is_a_normal_skip_and_is_not_retried(monkeypatch):
    window = _window(1)
    calls = []

    def fake_stage(_client, _model, prompt, schema):
        calls.append(prompt)
        return {"clip": None}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    assert _detail_call(
        monkeypatch, [window], [], min_clips=1
    ) == []
    assert len(calls) == 1


def test_fallback_keeps_small_global_ranking_headroom(monkeypatch):
    primary = [_window(1), _window(2)]
    fallback = [_window(3), _window(4), _window(5)]
    called_ids = []

    def fake_stage(_client, _model, prompt, schema):
        window_id = _ids_in_prompt(prompt)[0]
        called_ids.append(window_id)
        window = next(
            item for item in primary + fallback if item["id"] == window_id
        )
        if window in primary:
            return {"clip": None}, {"local": True, "total_cost": 0.0}
        return {"clip": _clip(window)}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    clips = _detail_call(
        monkeypatch, primary, fallback, min_clips=2
    )

    assert called_ids == [
        "window_001",
        "window_002",
        "window_003",
        "window_004",
        "window_005",
    ]
    assert [clip["source_window_id"] for clip in clips] == [
        "window_003",
        "window_004",
        "window_005",
    ]


def test_fallback_window_calls_are_bounded_by_configuration(monkeypatch):
    primary = [_window(1)]
    fallback = [_window(2), _window(3), _window(4), _window(5)]
    called_ids = []
    monkeypatch.setenv("LLM_DETAIL_FALLBACK_WINDOWS", "2")

    def fake_stage(_client, _model, prompt, schema):
        called_ids.append(_ids_in_prompt(prompt)[0])
        return {"clip": None}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    assert _detail_call(
        monkeypatch, primary, fallback, min_clips=3
    ) == []
    assert called_ids == ["window_001", "window_002", "window_003"]


def test_overlapping_fallback_does_not_satisfy_the_floor(monkeypatch):
    primary = [{
        "id": "window_001", "start": 0.0, "end": 90.0,
        "text": "primary", "score": 90,
    }]
    fallback = [
        {
            "id": "window_002", "start": 60.0, "end": 150.0,
            "text": "overlapping fallback", "score": 80,
        },
        {
            "id": "window_003", "start": 160.0, "end": 250.0,
            "text": "distinct fallback", "score": 70,
        },
        {
            "id": "window_004", "start": 260.0, "end": 350.0,
            "text": "extra global-rank choice", "score": 60,
        },
    ]
    called_ids = []

    def fake_stage(_client, _model, prompt, schema):
        window_id = _ids_in_prompt(prompt)[0]
        called_ids.append(window_id)
        window = next(
            item for item in primary + fallback if item["id"] == window_id
        )
        candidate = _clip(window)
        if window_id == "window_001":
            candidate["start"], candidate["end"] = 50.0, 80.0
        elif window_id == "window_002":
            candidate["start"], candidate["end"] = 65.0, 95.0
        return {"clip": candidate}, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    clips = _detail_call(
        monkeypatch, primary, fallback, min_clips=2
    )

    assert called_ids == [
        "window_001",
        "window_002",
        "window_003",
        "window_004",
    ]
    assert len(clips) == 4
    assert len(main.dedupe_overlapping(clips)) == 3


def test_floor_is_measured_after_word_boundary_snapping(monkeypatch):
    primary = [
        {"id": "window_001", "start": 0.0, "end": 90.0, "text": "a", "score": 90},
        {"id": "window_002", "start": 30.0, "end": 120.0, "text": "b", "score": 80},
    ]
    fallback = [
        {"id": "window_003", "start": 130.0, "end": 220.0, "text": "c", "score": 70},
    ]
    called_ids = []

    def fake_stage(_client, _model, prompt, schema):
        window_id = _ids_in_prompt(prompt)[0]
        called_ids.append(window_id)
        window = next(item for item in primary + fallback if item["id"] == window_id)
        candidate = _clip(window)
        if window_id == "window_002":
            candidate["start"], candidate["end"] = 32.6, 62.6
        return {"clip": candidate}, {"local": True, "total_cost": 0.0}

    snapped = {
        "window_001": (10.0, 40.4),
        # Raw clips overlap 7.4/30 seconds, just below the 25% duplicate bar.
        # A realistic 0.2-second snap on each edge crosses that bar.
        "window_002": (32.4, 62.6),
        "window_003": (140.0, 170.0),
    }

    def fake_snap(start, end, words, duration, source_window, **kwargs):
        return snapped[source_window["id"]]

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setattr(main, "snap_clip_to_words_in_window", fake_snap)

    clips = main._run_local_detail_windows(
        client=None,
        model_name="qwen2.5:7b",
        primary_windows=primary,
        fallback_windows=fallback,
        video_duration=1000.0,
        language="en",
        min_secs=15.0,
        max_secs=60.0,
        min_clips=2,
        max_clips=2,
        costs=[],
        words=[{"w": "x", "s": 0.0, "e": 0.1}],
    )

    assert called_ids == ["window_001", "window_002", "window_003"]
    assert len(main.dedupe_overlapping(clips)) == 2
    assert all(clip["_word_snapped"] is True for clip in clips)


def test_malformed_single_window_does_not_abort_later_windows(monkeypatch):
    primary = [_window(1), _window(2)]
    calls = []

    def fake_stage(_client, _model, prompt, schema):
        window_id = _ids_in_prompt(prompt)[0]
        calls.append(window_id)
        if window_id == "window_001":
            raise gemini_worker.ModelResponseError("not usable JSON")
        return {
            "clip": _clip(primary[1])
        }, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    clips = _detail_call(
        monkeypatch, primary, [], min_clips=1
    )

    assert calls == ["window_001", "window_002"]
    assert [clip["source_window_id"] for clip in clips] == ["window_002"]


def test_local_window_schema_accepts_an_explicit_no_clip():
    parsed = gemini_worker.validate_local_response_items(
        {"clip": None}, gemini_worker.LocalWindowDetailResponse
    )
    assert parsed == {"clip": None}


def test_local_window_schema_keeps_valid_boundaries_when_copy_is_missing():
    parsed = gemini_worker.validate_local_response_items(
        {"clip": {"start": 10, "end": 40}},
        gemini_worker.LocalWindowDetailResponse,
    )

    assert parsed["clip"]["start"] == 10
    assert parsed["clip"]["end"] == 40
    assert parsed["clip"]["video_title_for_youtube_short"] == ""
    assert parsed["clip"]["youtube_tags"] == []


def test_local_window_schema_sanitizes_bad_optional_copy_types():
    parsed = gemini_worker.validate_local_response_items(
        {
            "clip": {
                "start": 10,
                "end": 40,
                "predicted_score": 120,
                "video_title_for_youtube_short": 42,
                "youtube_tags": "one tag",
                "youtube_hashtags": [7, "#shorts"],
                "thumbnail_brief": None,
            }
        },
        gemini_worker.LocalWindowDetailResponse,
    )

    assert parsed["clip"]["start"] == 10
    assert parsed["clip"]["end"] == 40
    assert parsed["clip"]["predicted_score"] == 100
    assert parsed["clip"]["video_title_for_youtube_short"] == ""
    assert parsed["clip"]["youtube_tags"] == ["one tag"]
    assert parsed["clip"]["youtube_hashtags"] == ["#shorts"]
    assert parsed["clip"]["thumbnail_brief"]["subject"] == ""


def test_local_rank_schema_clamps_out_of_range_scores_before_reconciliation():
    parsed = gemini_worker.validate_local_response_items(
        {
            "candidates": [
                {"candidate_id": "candidate_001", "score": 999},
                {"candidate_id": "candidate_002", "score": -10},
            ]
        },
        gemini_worker.LocalClipRankResponse,
    )

    assert [item["score"] for item in parsed["candidates"]] == [100, 0]


def test_transport_or_context_error_remains_fatal(monkeypatch):
    def fake_stage(*_args, **_kwargs):
        raise RuntimeError("context window exceeded")

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    with pytest.raises(RuntimeError, match="context window exceeded"):
        _detail_call(monkeypatch, [_window(1)], [], min_clips=1)


def test_global_rank_reconciles_unknown_duplicate_and_missing_ids(monkeypatch):
    candidates = [
        _clip(_window(1), score=80),
        _clip(_window(2), score=70),
        _clip(_window(3), score=90),
    ]

    def fake_stage(_client, _model, prompt, schema):
        assert schema is gemini_worker.LocalClipRankResponse
        assert all(
            candidate_id in prompt
            for candidate_id in (
                "candidate_001",
                "candidate_002",
                "candidate_003",
            )
        )
        return {
            "candidates": [
                {"candidate_id": "candidate_001", "score": 40, "reason": "first"},
                {"candidate_id": "candidate_001", "score": 95, "reason": "better"},
                {"candidate_id": "candidate_002", "score": 999, "reason": "clamp"},
                {"candidate_id": "invented", "score": 100, "reason": "ignore"},
                # candidate_003 omitted: retain its existing score.
            ]
        }, {"local": True, "total_cost": 0.0}

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    ranked = main._rank_local_candidates(
        client=None,
        model_name="qwen2.5:7b",
        candidates=candidates,
        transcript_result={"segments": []},
        language="en",
        costs=[],
    )

    assert [item["_candidate_id"] for item in ranked] == [
        "candidate_002",
        "candidate_001",
        "candidate_003",
    ]
    assert [item["predicted_score"] for item in ranked] == [100, 95, 90]


def test_global_rank_failure_preserves_every_valid_candidate(monkeypatch):
    candidates = [
        _clip(_window(1), score=80),
        _clip(_window(2), score=90),
    ]

    def fake_stage(*_args, **_kwargs):
        raise RuntimeError("local rank request failed")

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)

    ranked = main._rank_local_candidates(
        client=None,
        model_name="qwen2.5:7b",
        candidates=candidates,
        transcript_result={"segments": []},
        language="en",
        costs=[],
    )

    assert len(ranked) == len(candidates)
    assert sorted(item["start"] for item in ranked) == sorted(
        item["start"] for item in candidates
    )
    assert [item["predicted_score"] for item in ranked] == [90, 80]


def test_local_prompts_keep_detail_and_ranking_payloads_compact():
    detail_prompt = gemini_worker.LOCAL_WINDOW_DETAIL_PROMPT_TEMPLATE
    rank_prompt = gemini_worker.LOCAL_CLIP_RANK_PROMPT_TEMPLATE

    assert re.search(
        r"(zero|0)[ -](?:or|to)[ -](?:one|1)", detail_prompt, re.I
    )
    assert "candidate_id" in rank_prompt
    # Global ranking must compare summaries, not repeat full transcript windows.
    assert "WINDOW_TRANSCRIPT" not in rank_prompt
    assert "windows_json" not in rank_prompt.lower()


def test_pure_global_rank_helper_never_drops_a_valid_candidate():
    candidates = [
        {**_clip(_window(1), score=60), "_candidate_id": "candidate_001"},
        {**_clip(_window(2), score=70), "_candidate_id": "candidate_002"},
        {**_clip(_window(3), score=80), "_candidate_id": "candidate_003"},
    ]
    rankings = [
        {"candidate_id": "candidate_001", "score": 10},
        {"candidate_id": "candidate_001", "score": 90},
        {"candidate_id": "candidate_002", "score": -100},
        {"candidate_id": "unknown", "score": 100},
    ]

    ranked = apply_global_candidate_ranking(candidates, rankings)

    assert [item["_candidate_id"] for item in ranked] == [
        "candidate_001",
        "candidate_003",
        "candidate_002",
    ]
    assert [item["predicted_score"] for item in ranked] == [90, 80, 0]


def test_overlap_deduplication_uses_the_global_score():
    earlier = {
        **_clip(_window(1), score=90),
        "_candidate_id": "candidate_001",
        "start": 10,
        "end": 40,
    }
    later = {
        **_clip(_window(1), score=60),
        "_candidate_id": "candidate_002",
        "start": 20,
        "end": 50,
    }

    ranked = apply_global_candidate_ranking(
        [earlier, later],
        [
            {"candidate_id": "candidate_001", "score": 70},
            {"candidate_id": "candidate_002", "score": 95},
        ],
    )
    deduped = main.dedupe_overlapping(ranked)

    assert [item["_candidate_id"] for item in deduped] == ["candidate_002"]


def test_global_overlap_choice_cannot_collapse_an_achieved_floor():
    candidates = [
        {
            **_clip(_window(1), score=90),
            "_candidate_id": "candidate_a",
            "_detail_predicted_score": 90,
            "start": 0,
            "end": 30,
        },
        {
            **_clip(_window(1), score=100),
            "_candidate_id": "candidate_b",
            "_detail_predicted_score": 70,
            "start": 20,
            "end": 50,
        },
        {
            **_clip(_window(1), score=80),
            "_candidate_id": "candidate_c",
            "_detail_predicted_score": 80,
            "start": 40,
            "end": 70,
        },
    ]

    selected, preserved = dedupe_ranked_with_floor(candidates, min_clips=2)

    assert preserved is True
    assert [item["_candidate_id"] for item in selected] == [
        "candidate_a",
        "candidate_c",
    ]
