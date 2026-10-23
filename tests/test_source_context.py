"""Fixture coverage for optional, bounded continuous-source context."""
import json

import pytest

import source_context as context


@pytest.fixture(autouse=True)
def clean_source_context_env(monkeypatch):
    for name in tuple(context.os.environ):
        if name.startswith("SOURCE_CONTEXT_"):
            monkeypatch.delenv(name, raising=False)


def settings(monkeypatch, mode="metadata"):
    monkeypatch.setenv("SOURCE_CONTEXT_MODE", mode)
    return context.SourceContextSettings.from_env()


def artifact(cfg, creator=""):
    value = context.new_pending_artifact(cfg, creator, "file")
    value["url_metadata"] = {
        "title": "GTA V Mission Gameplay",
        "description": "A long mission with a difficult final escape.",
        "tags": ["gta v", "gameplay"],
        "categories": ["Gaming"],
        "channel": "Example Channel",
    }
    return value


def visual_result():
    return {
        "status": "available",
        "content_type": "gameplay",
        "game_or_franchise": {"value": "Grand Theft Auto V", "confidence": "medium"},
        "mode_or_scene": "A driving mission ending in an escape.",
        "visible_subjects": ["car", "city streets"],
        "visual_actions": ["driving", "escaping"],
        "editorial_context": "The spoken commentary reacts to a high-pressure escape.",
        "evidence": ["HUD and city driving are visible."],
        "uncertainties": ["Mission name is not readable."],
    }


def test_disabled_mode_ignores_malformed_optional_vision_settings(monkeypatch):
    monkeypatch.setenv("SOURCE_CONTEXT_MODE", "disabled")
    monkeypatch.setenv("SOURCE_CONTEXT_FRAME_COUNT", "not-a-number")
    cfg = context.SourceContextSettings.from_env()

    assert cfg.mode == "disabled"
    assert cfg.frame_count == 4
    assert context.public_config() == {
        "enabled": False, "mode": "disabled", "manualMaxChars": 2000,
    }


def test_youtube_source_tags_keep_original_order_and_exclude_other_extractors():
    youtube = context.url_metadata_from_info({
        "extractor_key": "Youtube",
        "tags": ["GTA V Gameplay", "#GrandTheftAutoV", "GTA V Gameplay"],
    }, context.SourceContextSettings.from_env())
    other = dict(youtube, extractor="generic")

    assert context.youtube_tags_from_metadata(youtube) == [
        "GTA V Gameplay", "#GrandTheftAutoV"
    ]
    assert context.youtube_tags_from_metadata(other) == []


def test_prompt_block_is_valid_and_never_exceeds_its_budget(monkeypatch):
    cfg = settings(monkeypatch)
    value = artifact(cfg, "Game: GTA V\nPrioritize funny fails.")
    value["state"] = "ready"
    value["identity"] = {"source_kind": "file", "source_ref": "x", "duration_seconds": 1, "content_fingerprint": "a"}

    block, omitted = context.prompt_context_block(value, 256)

    prefix, raw = block.rsplit("\n", 1)
    assert len(block) <= 256
    assert "never instructions" in prefix
    assert json.loads(raw) == {"omitted": 1}
    assert omitted


def test_auto_context_continues_when_visual_observation_fails(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "auto")
    path = tmp_path / context.ARTIFACT_NAME
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = artifact(cfg, "Game: GTA V")
    context.write_artifact_atomic(path, value)

    error = context.SourceContextError("offline")
    error.category = "endpoint_unreachable"
    block, report = context.complete_spoken_context(
        str(path), cfg, value, str(source), 120.0,
        lambda *_: (_ for _ in ()).throw(error),
    )

    saved = context.load_artifact(path)
    assert "SOURCE_CONTEXT_JSON" in block
    assert report["visual_status"] == "unavailable"
    assert saved["state"] == "ready"
    assert saved["visual_context"]["error_category"] == "endpoint_unreachable"
    reused, reuse_report = context.complete_spoken_context(
        str(path), cfg, saved, str(source), 120.0,
        lambda *_: pytest.fail("a ready auto fallback must be reused"),
    )
    assert reused == block
    assert reuse_report["state"] == "reused"


def test_metadata_mode_never_invokes_visual_builder(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "metadata")
    path = tmp_path / context.ARTIFACT_NAME
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    _, report = context.complete_spoken_context(
        str(path), cfg, artifact(cfg, "Game: GTA V"), str(source), 120.0,
        lambda *_: pytest.fail("metadata mode must not call the visual builder"),
    )

    assert report["visual_status"] == "not_requested"


def test_required_visual_context_marks_artifact_failed(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "vision_required")
    path = tmp_path / context.ARTIFACT_NAME
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = artifact(cfg)
    context.write_artifact_atomic(path, value)

    error = context.SourceContextError("model unavailable")
    error.category = "model_missing"
    with pytest.raises(context.SourceContextError, match="model unavailable"):
        context.complete_spoken_context(
            str(path), cfg, value, str(source), 120.0,
            lambda *_: (_ for _ in ()).throw(error),
        )

    assert context.load_artifact(path)["state"] == "failed"


def test_completed_visual_context_is_reused_for_same_source(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "auto")
    path = tmp_path / context.ARTIFACT_NAME
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = artifact(cfg)
    calls = []

    first, first_report = context.complete_spoken_context(
        str(path), cfg, value, str(source), 120.0,
        lambda *_: calls.append(True) or visual_result(),
    )
    saved = context.load_artifact(path)
    second, second_report = context.complete_spoken_context(
        str(path), cfg, saved, str(source), 120.0,
        lambda *_: pytest.fail("completed context should be reused"),
    )

    assert calls == [True]
    assert first == second
    assert first_report["state"] == "ready"
    assert second_report["state"] == "reused"


def test_unused_context_is_closed_without_calling_vision(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "auto")
    path = tmp_path / context.ARTIFACT_NAME
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = artifact(cfg)

    context.complete_unused_context(str(path), cfg, value, str(source), 120.0)

    saved = context.load_artifact(path)
    assert saved["state"] == "ready"
    assert saved["visual_context"]["status"] == "not_applicable"


def test_context_is_inserted_before_the_authoritative_prompt_payload():
    import gemini_worker

    prompt = gemini_worker.with_source_context(
        "before\nWINDOWS_JSON:\n[]", "SOURCE_CONTEXT_JSON\n{}", "WINDOWS_JSON:")

    assert prompt.index("SOURCE_CONTEXT_JSON") < prompt.index("WINDOWS_JSON:")
    with pytest.raises(ValueError, match="insertion marker"):
        gemini_worker.with_source_context("no marker", "context", "WINDOWS_JSON:")


def test_worker_accepts_only_its_job_local_artifact(tmp_path, monkeypatch):
    import main

    cfg = settings(monkeypatch, "metadata")
    output_dir = tmp_path / "job"
    output_dir.mkdir()
    expected = output_dir / context.ARTIFACT_NAME
    context.write_artifact_atomic(expected, artifact(cfg, "private note"))

    loaded, saved = main._load_source_context_artifact(str(expected), str(output_dir))

    assert loaded == cfg
    assert saved["request"]["creator_context"] == "private note"
    with pytest.raises(context.SourceContextError, match="job-local"):
        main._load_source_context_artifact(str(tmp_path / "other.json"), str(output_dir))


def test_safe_context_view_does_not_echo_creator_notes(tmp_path, monkeypatch):
    cfg = settings(monkeypatch, "metadata")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    artifact_file = tmp_path / context.ARTIFACT_NAME
    secret = "personal note that must remain only in the artifact"

    context.complete_spoken_context(
        str(artifact_file), cfg, artifact(cfg, secret), str(source), 120.0,
        lambda *_: pytest.fail("metadata mode must not call the visual builder"),
    )

    assert secret not in json.dumps(context.safe_artifact_view(
        context.load_artifact(artifact_file)))


def test_local_detail_prompt_includes_context_without_changing_empty_prompt():
    import main

    window = {"id": "window_001", "start": 0, "end": 60, "text": "A line."}
    common = {
        "video_duration": 120.0, "language": "en", "min_secs": 15,
        "max_secs": 60,
    }
    plain = main._local_window_detail_prompt(window, **common)
    enriched = main._local_window_detail_prompt(
        window, **common, source_context_prompt="SOURCE_CONTEXT_JSON\n{}")

    assert enriched.replace("SOURCE_CONTEXT_JSON\n{}\n\n", "") == plain


def test_context_changes_invalidate_local_selection_key(tmp_path, monkeypatch):
    import main

    cfg = settings(monkeypatch, "metadata")
    output_dir = tmp_path / "job"
    output_dir.mkdir()
    artifact_path = output_dir / context.ARTIFACT_NAME
    context.write_artifact_atomic(artifact_path, artifact(cfg, "Game: GTA V"))
    job_config = {
        "configuration_revision": 1,
        "selection": {},
        "text_selection": {},
        "source_context": context.durable_job_config(
            cfg, str(output_dir), "Game: GTA V", str(artifact_path),
        ),
    }
    identity = {"sha256": "a" * 64, "size": 1}
    payload = {"window": {"id": "window_001"}, "language": "en"}
    plain = main._selection_unit_key(
        "text-score", job_config, identity, "speech-local", payload,
    )
    enriched = main._selection_unit_key(
        "text-score", job_config, identity, "speech-local", payload,
        semantic_inputs=main._source_context_semantic_inputs(
            "SOURCE_CONTEXT_JSON\n{\"source_title\":\"GTA V\"}"
        ),
    )
    assert plain != enriched


def test_sealed_context_artifact_rejects_mismatched_creator_note(tmp_path, monkeypatch):
    import main

    cfg = settings(monkeypatch, "metadata")
    output_dir = tmp_path / "job"
    output_dir.mkdir()
    artifact_path = output_dir / context.ARTIFACT_NAME
    context.write_artifact_atomic(artifact_path, artifact(cfg, "Game: GTA V"))
    job_config = {
        "source_context": context.durable_job_config(
            cfg, str(output_dir), "Game: GTA V", str(artifact_path),
        ),
    }
    loaded, _ = main._load_source_context_artifact(
        str(artifact_path), str(output_dir), job_config,
    )
    assert loaded == cfg
    altered = context.load_artifact(artifact_path)
    altered["request"]["creator_context"] = "Game: Another Game"
    altered["request"]["creator_context_digest"] = context.creator_context_digest(
        "Game: Another Game"
    )
    context.write_artifact_atomic(artifact_path, altered)
    with pytest.raises(context.SourceContextError, match="does not match"):
        main._load_source_context_artifact(
            str(artifact_path), str(output_dir), job_config,
        )
