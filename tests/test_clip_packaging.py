import json

import clip_packaging as packaging


def test_normalise_metadata_never_rejects_an_otherwise_valid_clip():
    clip = packaging.normalise_clip_packaging_metadata({
        "video_title_for_youtube_short": "A clip",
        "youtube_description_intro": "Hello\x00 <world>\n\n\nSecond paragraph",
        "youtube_tags": ["#Gaming", "gaming", 123, "Boss Fight"],
        "youtube_hashtags": [" shorts ", "#GAMING", "#gaming", None],
        "thumbnail_brief": {
            "subject": "Hero\x00",
            "thumbnail_text": "A" * 80,
            "must_include": ["sword", "SWORD", 1],
        },
    })

    assert clip["youtube_description_intro"] == "Hello world\n\nSecond paragraph"
    assert clip["youtube_tags"] == ["Gaming", "Boss Fight"]
    assert clip["youtube_hashtags"] == ["#shorts", "#GAMING"]
    assert clip["thumbnail_brief"]["subject"] == "Hero"
    assert len(clip["thumbnail_brief"]["thumbnail_text"]) == 40
    assert clip["thumbnail_brief"]["must_include"] == ["sword"]


def test_channel_profile_stays_confined_and_metadata_is_bounded(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(packaging, "__file__", str(tmp_path / "clip_packaging.py"))
    (config_dir / "youtube-channel-profile.json").write_text(json.dumps({
        "default_tags": ["channel"],
        "default_hashtags": ["#channel"],
        "description_sections": ["z" * 1500] * 12,
    }), encoding="utf-8")
    monkeypatch.setenv("YOUTUBE_CHANNEL_PROFILE_PATH", str(config_dir / "youtube-channel-profile.json"))

    description, tags, truncated, warning = packaging.compose_youtube_metadata({
        "youtube_description_intro": "intro",
        "youtube_tags": ["clip"],
        "youtube_hashtags": ["#clip"],
    })

    assert tags == ["clip", "channel"]
    assert truncated is True
    assert warning is None
    assert len(description.encode("utf-8")) <= packaging.DESCRIPTION_LIMIT_BYTES

    monkeypatch.setenv("YOUTUBE_CHANNEL_PROFILE_PATH", str(tmp_path / "outside.json"))
    profile, warning = packaging.load_youtube_channel_profile()
    assert profile == {"default_tags": [], "default_hashtags": [], "description_sections": []}
    assert "inside OpenShorts/config" in warning


def test_clip_transcript_excerpt_uses_only_overlapping_clip_words():
    transcript = {
        "segments": [{
            "words": [
                {"word": "Before", "start": 0, "end": 1},
                {"word": "Keep", "start": 1.5, "end": 2},
                {"word": "this", "start": 2, "end": 2.5},
                {"word": "After", "start": 3.1, "end": 4},
            ]
        }]
    }

    assert packaging.clip_transcript_excerpt(transcript, 1.75, 3) == "Keep this"
