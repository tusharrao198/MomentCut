"""Regression coverage for local transcript score reconciliation."""

from clip_selection import reconcile_scored_windows


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


def test_score_reconciliation_keeps_every_input_window_once():
    source = _windows(3)
    scored = [
        {"id": "window_002", "start": -99, "end": 999, "score": 22, "reason": "weak"},
        {"id": "window_002", "score": 91, "reason": "stronger duplicate"},
        {"id": "invented", "score": 100, "reason": "must be ignored"},
        {"id": ["not", "an", "id"], "score": 100, "reason": "must be ignored"},
        {"id": "window_003", "score": "not-a-number", "reason": "bad score"},
    ]

    reconciled = reconcile_scored_windows(scored, source)

    assert [item["id"] for item in reconciled] == [
        "window_001", "window_002", "window_003"
    ]
    assert [item["score"] for item in reconciled] == [0, 91, 0]
    assert reconciled[1]["start"] == source[1]["start"]
    assert reconciled[1]["end"] == source[1]["end"]
