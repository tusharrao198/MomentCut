import checkpoint_store
import burned_subtitles


def _store(tmp_path):
    config = {
        "schema_version": 1,
        "job_id": "analysis-job",
        "configuration_revision": 1,
        "source": {"kind": "upload", "retained_relative_path": "source.mp4"},
        "pipeline": {},
        "checkpointing_enabled": True,
    }
    identity = {
        "sha256": checkpoint_store.sha256_bytes(b"source"),
        "size": 6,
        "duration_ms": 1000,
        "fps": 30.0,
    }
    return checkpoint_store.CheckpointStore(
        str(tmp_path), config, True, 1, identity,
    )


def test_explicit_none_source_analysis_is_a_cache_hit(tmp_path):
    store = _store(tmp_path)
    key = checkpoint_store.compatibility_key("source-subtitles", {
        "source_content_identity": store.source_identity,
        "route": "gemini",
        "resolved_stage_config": {"minimum_presence": 0.4},
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": ["a" * 64],
    })
    payload = {
        "outcome": "none",
        "band": None,
        "sample_hashes": ["a" * 64],
        "config": {"model": "gemini-model"},
        "implementation_version": "source-subtitles-v1",
    }
    assert store.save_fixed("source-subtitles", key, payload, {})
    assert store.load_fixed("source-subtitles", key)["payload"] == payload


def test_completed_none_layout_is_a_cache_hit(tmp_path):
    store = _store(tmp_path)
    key = checkpoint_store.compatibility_key("source-layout", {
        "source_content_identity": store.source_identity,
        "route": "source-layout",
        "resolved_stage_config": {"mode": "1"},
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": ["a" * 64],
    })
    payload = {
        "outcome": "complete",
        "reason": None,
        "layout": "none",
        "applied_flags": ["split_layout", "active_speaker"],
        "sample_hashes": ["a" * 64],
        "config": {"model": "gemini-model"},
        "implementation_version": "source-layout-v1",
    }
    assert store.save_fixed("source-layout", key, payload, {})
    assert store.load_fixed("source-layout", key)["payload"] == payload


def test_failed_layout_fallback_is_not_checkpointed(tmp_path):
    store = _store(tmp_path)
    key = checkpoint_store.compatibility_key("source-layout", {
        "source_content_identity": store.source_identity,
        "route": "source-layout",
        "resolved_stage_config": {"mode": "1"},
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": ["a" * 64],
    })
    assert not store.save_fixed(
        "source-layout",
        key,
        {
            "outcome": "failure",
            "reason": "missing_api_key",
            "layout": "none",
            "applied_flags": ["split_layout", "active_speaker"],
            "sample_hashes": ["a" * 64],
            "config": {"model": "gemini-model"},
            "implementation_version": "source-layout-v1",
        },
        {},
    )


def test_subtitle_analysis_honours_gemini_analysis_kill_switch(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    monkeypatch.setattr(
        "gemini_feature_flags.clip_analysis_enabled",
        lambda: False,
    )

    result = burned_subtitles.detect_with_provenance(
        "unused.mp4",
        frames=[b"not-sent"],
        config={
            "enabled": True,
            "model": "gemini-model",
            "minimum_presence": 0.4,
            "highest_band": 0.75,
            "lowest_band": 0.97,
            "margin": 0.02,
            "sample_frames": 1,
            "sample_width": 640,
            "prompt_version": "burned-subtitles-v1",
        },
    )

    assert result["outcome"] == "failure"
    assert result["reason"] == "analysis_disabled"
    assert result["sample_hashes"] == []


def test_scene_fallback_provenance_is_preserved(tmp_path):
    store = _store(tmp_path)
    key = checkpoint_store.compatibility_key("source-scenes", {
        "source_content_identity": store.source_identity,
        "route": "transnetv2",
        "resolved_stage_config": {},
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": [],
    })
    detector = {
        "requested_engine": "transnetv2",
        "effective_engine": "pyscenedetect",
        "fallback_reason": "RuntimeError",
    }
    assert store.save_fixed(
        "source-scenes",
        key,
        {"boundaries_seconds": [0.0, 1.0], "fps": 30.0,
         "detector": detector},
        {"effective_detector": detector},
    )
    assert store.load_fixed("source-scenes", key)["payload"]["detector"][
        "effective_engine"
    ] == "pyscenedetect"


def test_malformed_source_analysis_payload_is_not_persisted(tmp_path):
    store = _store(tmp_path)
    key = checkpoint_store.compatibility_key("source-scenes", {
        "source_content_identity": store.source_identity,
        "route": "pyscenedetect",
        "resolved_stage_config": {},
        "expected_upstream_compatibility_keys": [],
        "semantic_input_hashes": [],
    })
    assert not store.save_fixed(
        "source-scenes",
        key,
        {
            "boundaries_seconds": [0.0, "not-a-timestamp"],
            "fps": 30.0,
            "detector": {"effective_engine": "pyscenedetect"},
        },
        {},
    )
