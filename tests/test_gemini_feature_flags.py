import gemini_feature_flags as flags


def test_clip_ai_thumbnails_are_opt_in(monkeypatch):
    monkeypatch.delenv("CLIP_AI_THUMBNAILS_ENABLED", raising=False)
    assert flags.clip_ai_thumbnails_enabled() is False

    monkeypatch.setenv("CLIP_AI_THUMBNAILS_ENABLED", "1")
    assert flags.clip_ai_thumbnails_enabled() is True


def test_gemini_clip_analysis_keeps_legacy_default(monkeypatch):
    monkeypatch.delenv("GEMINI_CLIP_ANALYSIS_ENABLED", raising=False)
    assert flags.clip_analysis_enabled() is True
