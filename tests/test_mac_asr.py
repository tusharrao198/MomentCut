import sys
from types import SimpleNamespace

import pytest

import mac_asr


SMALL_REVISION = "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e"


def test_small_alias_resolves_to_benchmarked_snapshot(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.delenv("WHISPER_MLX_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MLX_MODEL_REVISION", raising=False)
    assert mac_asr.resolve_whisper_model() == (
        "mlx-community/whisper-small-mlx", SMALL_REVISION)


def test_custom_repository_requires_immutable_revision(monkeypatch):
    monkeypatch.setenv("WHISPER_MLX_MODEL", "owner/custom-whisper")
    monkeypatch.delenv("WHISPER_MLX_MODEL_REVISION", raising=False)
    with pytest.raises(mac_asr.MacASRConfigurationError):
        mac_asr.resolve_whisper_model()


def test_unknown_alias_is_not_guessed(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.delenv("WHISPER_MLX_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MLX_MODEL_REVISION", raising=False)
    with pytest.raises(mac_asr.MacASRConfigurationError):
        mac_asr.resolve_whisper_model()


def test_mlx_transcript_is_normalised_without_importing_at_module_load(
        monkeypatch, tmp_path):
    snapshot = tmp_path / SMALL_REVISION
    snapshot.mkdir()
    calls = {}

    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.delenv("WHISPER_MLX_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MLX_MODEL_REVISION", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )

    def transcribe(audio, **kwargs):
        calls.update({"audio": audio, **kwargs})
        return {
            "text": "Hello world.",
            "language": "en",
            "segments": [{
                "start": 0,
                "end": 1.25,
                "text": "Hello world.",
                "words": [
                    {"word": "Hello", "start": 0, "end": 0.4},
                    {"word": " world", "start": 0.4, "end": 0.9},
                    {"word": ".", "start": 0.9, "end": 1.0},
                ],
            }],
        }

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    result = mac_asr.transcribe_whisper(
        "clip.mp4",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    assert calls["path_or_hf_repo"] == str(snapshot)
    assert calls["beam_size"] == 5
    assert "vad_filter" not in calls
    assert result["segments"][0]["words"] == [
        {"word": " Hello", "start": 0.0, "end": 0.4},
        {"word": " world.", "start": 0.4, "end": 1.0},
    ]


def test_snapshot_identity_must_match_requested_revision(monkeypatch, tmp_path):
    wrong = tmp_path / ("b" * 40)
    wrong.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(wrong)),
    )
    with pytest.raises(mac_asr.MacASRConfigurationError):
        mac_asr._snapshot_path("owner/model", SMALL_REVISION)


def test_release_drops_model_and_clears_loaded_mlx_cache(monkeypatch):
    holder = SimpleNamespace(model=object(), model_path="/snapshot")
    cleared = []
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper.transcribe",
        SimpleNamespace(ModelHolder=holder),
    )
    monkeypatch.setitem(
        sys.modules, "mlx.core", SimpleNamespace(clear_cache=lambda: cleared.append(True)))

    assert mac_asr.release_whisper_model() is True
    assert holder.model is None and holder.model_path is None
    assert cleared == [True]
