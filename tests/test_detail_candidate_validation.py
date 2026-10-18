"""Regression coverage for local transcript detail candidate validation."""

from clip_selection import (
    normalise_detail_candidates,
    snap_clip_to_words_in_window,
)


def test_detail_candidates_are_clamped_then_rejected_when_not_renderable():
    source = [
        {"id": "window_a", "start": 10.0, "end": 75.0, "text": "a"},
        {"id": "window_b", "start": 80.0, "end": 120.0, "text": "b"},
    ]
    candidates = [
        {"source_window_id": "window_a", "start": 5.0, "end": 70.0},
        {"source_window_id": "window_b", "start": 85.0, "end": 115.0},
        {"source_window_id": "invented", "start": 10.0, "end": 40.0},
        {"source_window_id": ["not", "an", "id"], "start": 10.0, "end": 40.0},
        {"source_window_id": "window_a", "start": 70.0, "end": 40.0},
        {"source_window_id": "window_b", "start": 100.0, "end": 110.0},
        {"source_window_id": "window_a", "start": 10.0, "end": 75.0},
    ]

    clips = normalise_detail_candidates(
        candidates, source, video_duration=110.0, min_duration=15.0,
        max_duration=60.0)

    assert [(clip["source_window_id"], clip["start"], clip["end"]) for clip in clips] == [
        ("window_a", 10.0, 70.0),
        ("window_b", 85.0, 110.0),
    ]


def test_word_snap_does_not_escape_the_grounding_window():
    snapped = snap_clip_to_words_in_window(
        10.1,
        40.0,
        [
            {"w": "previous", "s": 9.0, "e": 9.5},
            {"w": "ending", "s": 39.0, "e": 40.0},
        ],
        video_duration=120.0,
        source_window={"id": "window_a", "start": 10.0, "end": 75.0},
        min_duration=15.0,
        max_duration=60.0,
    )

    assert snapped == (10.1, 40.0)
