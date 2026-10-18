"""Unit tests for the opt-in local Ollama silent-video picker."""
import json

import httpx
import pytest

import vision_backend as vision


@pytest.fixture(autouse=True)
def clean_vision_environment(monkeypatch):
    """Do not let a developer's local .env alter bounded-config assertions."""
    for name in (
        "VISION_PROVIDER",
        "VISION_BASE_URL",
        "VISION_MODEL",
        "VISION_TIMEOUT_SECONDS",
        "VISION_RETRY_ATTEMPTS",
        "VISION_WINDOW_SECONDS",
        "VISION_WINDOW_OVERLAP_SECONDS",
        "VISION_FRAMES_PER_WINDOW",
        "VISION_MAX_FRAME_WIDTH",
        "VISION_NUM_CTX",
        "VISION_CANDIDATES_PER_WINDOW",
        "VISION_SCENE_SNAP_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def local_vision(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "ollama")
    monkeypatch.setenv("VISION_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("VISION_MODEL", "qwen2.5vl:3b")
    return vision._config(60)


def _clip(start=10, end=40, score=80):
    return {
        "shorts": [{
            "start": start,
            "end": end,
            "predicted_score": score,
            "video_description_for_tiktok": "A striking visual reveal.",
            "video_description_for_instagram": "A striking visual reveal.",
            "video_title_for_youtube_short": "A striking visual reveal",
            "viral_hook_text": "Watch the reveal",
        }]
    }


def _ollama_response(payload):
    return httpx.Response(200, json={
        "message": {"role": "assistant", "content": json.dumps(payload)}
    })


def test_rejects_remote_endpoint_before_any_http(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "ollama")
    monkeypatch.setenv("VISION_BASE_URL", "https://example.com")
    with pytest.raises(vision.LocalVisionError, match="approved local endpoint"):
        vision._config(60)


def test_accepts_docker_host_endpoint(local_vision):
    assert local_vision.base_url == "http://host.docker.internal:11434"


def test_default_frame_request_uses_the_conservative_12gb_profile(local_vision):
    assert (local_vision.frames_per_window, local_vision.max_frame_width,
            local_vision.num_ctx) == (4, 480, 8192)


def test_blank_model_uses_the_approved_default(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "ollama")
    monkeypatch.setenv("VISION_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("VISION_MODEL", "")
    assert vision._config(60).model == "qwen2.5vl:3b"


def test_clamps_tunables_and_rejects_window_that_cannot_cover_a_clip(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "ollama")
    monkeypatch.setenv("VISION_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("VISION_TIMEOUT_SECONDS", "9999")
    monkeypatch.setenv("VISION_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("VISION_WINDOW_SECONDS", "999")
    monkeypatch.setenv("VISION_WINDOW_OVERLAP_SECONDS", "1")
    monkeypatch.setenv("VISION_FRAMES_PER_WINDOW", "-3")
    monkeypatch.setenv("VISION_MAX_FRAME_WIDTH", "2000")
    monkeypatch.setenv("VISION_NUM_CTX", "1")
    monkeypatch.setenv("VISION_CANDIDATES_PER_WINDOW", "20")
    monkeypatch.setenv("VISION_SCENE_SNAP_SECONDS", "9")

    config = vision._config(60)
    assert config.timeout_seconds == 600
    assert config.retry_attempts == 1
    assert (config.window_seconds, config.overlap_seconds) == (120, 60)
    assert config.frames_per_window == 4
    assert config.max_frame_width == 768
    assert config.num_ctx == 2048
    assert config.candidates_per_window == 3
    assert config.scene_snap_seconds == 3

    monkeypatch.setenv("VISION_WINDOW_SECONDS", "60")
    with pytest.raises(vision.LocalVisionError, match="greater than CLIP_MAX_SECONDS"):
        vision._config(60)


def test_dropped_frame_drops_its_matching_timestamp(monkeypatch):
    encoded = iter(("first", None, "third"))
    monkeypatch.setattr(vision, "_jpeg_base64", lambda _frame, _width: next(encoded))

    timestamps, images = vision._encode_samples(
        [1.0, 2.0, 3.0],
        [30, 60, 90],
        [object(), object(), object()],
        fps=30,
        duration=10,
        max_width=640,
    )

    assert timestamps == [1.0, 3.0]
    assert images == ["first", "third"]


def test_overlapping_windows_cover_every_maximum_length_candidate():
    windows = vision._analysis_windows(720, 120, 60)
    assert len(windows) == 11
    for start in range(0, 661):
        end = start + 60
        assert any(window_start <= start and window_end >= end
                   for window_start, window_end in windows)


def test_low_regular_frame_count_still_adds_at_most_two_scene_frames():
    samples = vision._sample_times(0, 120, list(range(1, 120)), 4)
    assert len(samples) == 6


def test_scene_snap_only_applies_when_duration_remains_valid():
    assert vision._snap_to_scenes(
        10.9, 40.2, [11.0, 41.0], 15, 60, 2
    ) == (11.0, 41.0)

    # Snapping the start from 10 to 12 would turn this 15-second clip into
    # a 13-second clip, so the original valid timestamp stays in place.
    assert vision._snap_to_scenes(
        10.0, 25.0, [12.0], 15, 60, 3
    ) == (10.0, 25.0)


def test_short_source_can_produce_one_full_duration_candidate():
    result = vision._normalise_window_shorts(
        _clip(start=1, end=9)["shorts"],
        window_start=0,
        window_end=10,
        source_duration=10,
        min_seconds=15,
        max_seconds=60,
        scene_boundaries=[],
        snap_seconds=2,
        candidates_per_window=2,
    )
    assert [(clip["start"], clip["end"]) for clip in result] == [(0.0, 10)]


def test_bad_json_gets_exactly_one_corrective_retry(local_vision, monkeypatch):
    calls = []

    class Client:
        def post(self, _url, json):
            calls.append(json)
            if len(calls) == 1:
                return httpx.Response(200, json={"message": {"content": "not json"}})
            return _ollama_response(_clip())

    result = vision._window_response(
        Client(), local_vision, "prompt", ["frame"], "Local vision window 0s–60s"
    )
    assert result["shorts"][0]["predicted_score"] == 80
    assert len(calls) == 2
    assert "prior response was invalid" in calls[1]["messages"][0]["content"]


def test_two_bad_json_replies_fail_after_one_corrective_retry(local_vision):
    calls = {"count": 0}

    class Client:
        def post(self, _url, json):
            calls["count"] += 1
            return httpx.Response(200, json={"message": {"content": "not json"}})

    with pytest.raises(vision.LocalVisionError, match="after one correction attempt"):
        vision._window_response(
            Client(), local_vision, "prompt", ["frame"], "Local vision window 0s–60s"
        )
    assert calls["count"] == 2


def test_503_retries_are_bounded(local_vision, monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(vision.time, "sleep", lambda *_: None)

    class Client:
        def post(self, *_args, **_kwargs):
            calls["count"] += 1
            return httpx.Response(503, text="model still loading")

    with pytest.raises(vision.LocalVisionError, match="after 3 attempt"):
        vision._post_with_retries(Client(), local_vision, "/api/chat", {}, "window")
    assert calls["count"] == 3


def test_401_fails_without_retry(local_vision):
    calls = {"count": 0}

    class Client:
        def post(self, *_args, **_kwargs):
            calls["count"] += 1
            return httpx.Response(401, text="unauthorized")

    with pytest.raises(vision.LocalVisionError, match="configuration failed"):
        vision._post_with_retries(Client(), local_vision, "/api/show", {}, "model")
    assert calls["count"] == 1


def test_one_failed_window_fails_the_job_and_attempts_unload(local_vision, monkeypatch):
    class Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(vision.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(vision, "_client", lambda **_kwargs: Client())
    monkeypatch.setattr(vision, "_preflight_model", lambda *_args: None)
    monkeypatch.setattr(vision, "_scene_boundaries", lambda *_args: [])
    monkeypatch.setattr(
        vision, "_decode_samples",
        lambda *_args: ([10.0], ["image"]),
    )
    responses = iter((_clip(), vision.LocalVisionError("second window timed out")))
    monkeypatch.setattr(
        vision, "_window_response",
        lambda *_args: next(responses),
    )
    unloaded = []
    monkeypatch.setattr(vision, "_unload", lambda *_args: unloaded.append(True))

    with pytest.raises(vision.LocalVisionError, match="second window timed out"):
        vision.select_clips(
            "source.mp4", 180, "en", 1, 3, 15, 60
        )
    assert unloaded == [True]


def test_empty_window_is_skipped_when_a_later_window_has_candidates(
        local_vision, monkeypatch):
    class Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(vision.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(vision, "_client", lambda **_kwargs: Client())
    monkeypatch.setattr(vision, "_preflight_model", lambda *_args: None)
    monkeypatch.setattr(vision, "_scene_boundaries", lambda *_args: [])
    monkeypatch.setattr(
        vision, "_decode_samples",
        lambda *_args: ([10.0], ["image"]),
    )
    responses = iter(({"shorts": []}, _clip(70, 100)))
    monkeypatch.setattr(
        vision, "_window_response",
        lambda *_args: next(responses),
    )
    unloaded = []
    monkeypatch.setattr(vision, "_unload", lambda *_args: unloaded.append(True))

    result = vision.select_clips("source.mp4", 180, "en", 1, 3, 15, 60)

    assert [(clip["start"], clip["end"]) for clip in result["shorts"]] == [
        (70.0, 100.0)
    ]
    assert unloaded == [True]


def test_all_empty_windows_fail_after_every_window_is_examined(
        local_vision, monkeypatch):
    class Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(vision.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(vision, "_client", lambda **_kwargs: Client())
    monkeypatch.setattr(vision, "_preflight_model", lambda *_args: None)
    monkeypatch.setattr(vision, "_scene_boundaries", lambda *_args: [])
    monkeypatch.setattr(
        vision, "_decode_samples",
        lambda *_args: ([10.0], ["image"]),
    )
    calls = {"count": 0}

    def empty_response(*_args):
        calls["count"] += 1
        return {"shorts": []}

    monkeypatch.setattr(vision, "_window_response", empty_response)
    unloaded = []
    monkeypatch.setattr(vision, "_unload", lambda *_args: unloaded.append(True))

    with pytest.raises(
            vision.LocalVisionError,
            match="Local Ollama vision returned no usable clip candidates"):
        vision.select_clips("source.mp4", 180, "en", 1, 3, 15, 60)

    assert calls["count"] == 2
    assert unloaded == [True]


def test_decode_failure_names_the_failed_window(local_vision, monkeypatch):
    class Capture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(vision.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(vision, "_client", lambda **_kwargs: Client())
    monkeypatch.setattr(vision, "_preflight_model", lambda *_args: None)
    monkeypatch.setattr(vision, "_scene_boundaries", lambda *_args: [])
    monkeypatch.setattr(
        vision, "_decode_samples",
        lambda *_args: (_ for _ in ()).throw(
            vision.LocalVisionError("could not decode any frame")),
    )
    monkeypatch.setattr(vision, "_unload", lambda *_args: None)

    with pytest.raises(
            vision.LocalVisionError,
            match=r"Local vision window 0\.000s–120\.000s: could not decode any frame"):
        vision.select_clips("source.mp4", 180, "en", 1, 3, 15, 60)


def test_duplicate_candidates_keep_the_higher_score():
    candidates = _clip(10, 40, 50)["shorts"] + _clip(12, 42, 90)["shorts"]
    deduped = vision.dedupe_overlapping(candidates)
    assert len(deduped) == 1
    assert deduped[0]["predicted_score"] == 90


def test_main_routes_silent_video_to_local_vision(monkeypatch):
    """The routing shim stays tiny: local opt-in must precede Gemini."""
    main = pytest.importorskip("main")
    seen = {}
    monkeypatch.setattr(main.vision_backend, "active", lambda: True)
    monkeypatch.setattr(main, "clip_duration_bounds", lambda: (15.0, 60.0))
    monkeypatch.setattr(
        main.vision_backend,
        "select_clips",
        lambda **kwargs: seen.update(kwargs) or {"shorts": []},
    )

    result = main.get_visual_clips("source.mp4", 120.0, "en")

    assert result == {"shorts": []}
    assert seen == {
        "video_path": "source.mp4",
        "video_duration": 120.0,
        "language": "en",
        "min_clips": 3,
        "max_clips": 15,
        "min_seconds": 15.0,
        "max_seconds": 60.0,
    }
