import sys
from types import SimpleNamespace

import pytest

import mac_asr


SMALL_REVISION = "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e"
PARAKEET_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"


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


def test_parakeet_default_resolves_to_benchmarked_snapshot(monkeypatch):
    monkeypatch.delenv("PARAKEET_MLX_MODEL", raising=False)
    monkeypatch.delenv("PARAKEET_MLX_MODEL_REVISION", raising=False)
    assert mac_asr.resolve_parakeet_model() == (
        "mlx-community/parakeet-tdt-0.6b-v3", PARAKEET_REVISION)


def test_custom_parakeet_repository_requires_immutable_revision(monkeypatch):
    monkeypatch.setenv("PARAKEET_MLX_MODEL", "owner/custom-parakeet")
    monkeypatch.delenv("PARAKEET_MLX_MODEL_REVISION", raising=False)
    with pytest.raises(mac_asr.MacASRConfigurationError):
        mac_asr.resolve_parakeet_model()


def test_failed_mlx_import_is_not_retried_in_the_same_process(monkeypatch):
    calls = []

    def fail(name):
        calls.append(name)
        raise ImportError("No Metal device available")

    monkeypatch.setattr(mac_asr, "_mlx_import_error", None)
    monkeypatch.setattr(mac_asr.importlib, "import_module", fail)

    with pytest.raises(RuntimeError, match="parakeet_mlx is unavailable"):
        mac_asr._import_mlx_package("parakeet_mlx")
    with pytest.raises(RuntimeError, match="initialization previously failed"):
        mac_asr._import_mlx_package("mlx_whisper")
    assert calls == ["parakeet_mlx"]


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


def test_parakeet_transcript_is_normalised_to_openshorts_contract(
        monkeypatch, tmp_path):
    snapshot = tmp_path / PARAKEET_REVISION
    snapshot.mkdir()
    calls = {}
    tokens = [
        SimpleNamespace(text="Hello", start=0, end=0.4),
        SimpleNamespace(text=" world", start=0.4, end=0.9),
        SimpleNamespace(text=".", start=0.9, end=1.0),
    ]
    result = SimpleNamespace(
        text="Hello world.",
        sentences=[SimpleNamespace(text="Hello world.", tokens=tokens)],
    )

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.update({"audio": audio, **kwargs})
            return result

    monkeypatch.delenv("PARAKEET_MLX_MODEL", raising=False)
    monkeypatch.delenv("PARAKEET_MLX_MODEL_REVISION", raising=False)
    monkeypatch.setattr(
        mac_asr, "_snapshot_path",
        lambda repository, revision: str(snapshot))
    monkeypatch.setattr(mac_asr, "_parakeet_model", None)
    monkeypatch.setattr(mac_asr, "_parakeet_key", None)
    monkeypatch.setitem(
        sys.modules,
        "parakeet_mlx",
        SimpleNamespace(from_pretrained=lambda path: Model()),
    )

    transcript = mac_asr.transcribe_parakeet("clip.wav")

    assert calls["audio"] == "clip.wav"
    assert calls["chunk_duration"] == 120
    assert calls["overlap_duration"] == 15
    assert transcript == {
        "text": "Hello world.",
        "segments": [{
            "start": 0.0,
            "end": 1.0,
            "text": "Hello world.",
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.4},
                {"word": " world.", "start": 0.4, "end": 1.0},
            ],
        }],
    }


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


def test_release_models_drops_whisper_and_parakeet_before_clearing_cache(
        monkeypatch):
    holder = SimpleNamespace(model=object(), model_path="/whisper")
    cleared = []
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper.transcribe",
        SimpleNamespace(ModelHolder=holder),
    )
    monkeypatch.setitem(
        sys.modules, "mlx.core",
        SimpleNamespace(clear_cache=lambda: cleared.append(True)),
    )
    monkeypatch.setattr(mac_asr, "_parakeet_model", object())
    monkeypatch.setattr(mac_asr, "_parakeet_key", ("repo", PARAKEET_REVISION))

    assert mac_asr.release_models() is True
    assert holder.model is None and holder.model_path is None
    assert mac_asr._parakeet_model is None
    assert mac_asr._parakeet_key is None
    assert cleared == [True]
