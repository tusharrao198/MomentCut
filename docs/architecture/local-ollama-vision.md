# Local Ollama vision: implementation design

This document describes the code architecture for local silent-video clip
selection. It is for maintainers and future feature work.

For the complete Windows installation, `.env` configuration, Docker commands,
GPU verification, and recovery steps, use
[`../runbooks/local-ollama-windows-setup.md`](../runbooks/local-ollama-windows-setup.md).

## Scope and routing

The feature replaces only the Gemini-based moment-selection step for a video
without meaningful spoken content. It does not change Whisper, captioning,
rendering, or the normal transcript-driven picker.

| Source video | Detection | Selection path |
|---|---|---|
| Meaningful spoken audio | Whisper has enough words | Existing text picker; `qwen2.5:7b` when `LLM_BASE_URL` is configured, otherwise Gemini. |
| No audio stream | Existing ffprobe check raises `NoAudioError` before Whisper | `qwen2.5vl:3b` through Ollama when `VISION_PROVIDER=ollama`, otherwise Gemini vision. |
| Audio track but little/no useful speech | Whisper result fails `speech_is_sparse()` | Same visual route as a no-audio video. |

```text
                   input video
                        |
              ffprobe: audio stream?
                   /             \
                 no               yes
                 |                 |
      local vision selector      Whisper
      or Gemini fallback          |
                         enough meaningful words?
                              /             \
                            yes              no
                            |                 |
                 transcript picker       visual selector
```

`VISION_PROVIDER=ollama` is an explicit privacy opt-in. If it is enabled, a
local failure raises `LocalVisionError`; the code must not silently fall back to
Gemini and upload footage.

## Implemented files

| File | Responsibility |
|---|---|
| `vision_backend.py` | Local native-Ollama adapter: endpoint validation, frame sampling, VLM calls, output validation, bounded retries, deduplication, and VLM unload. |
| `main.py` | Calls the adapter first from `get_visual_clips()` when local vision is enabled; preserves the existing Gemini path otherwise. |
| `dashboard/src/App.jsx` | Does not block Clip Generator on an Upload-Post key; displays local results as `LOCAL · <model>`. |
| `docker-compose.yml` | Keeps self-host ports loopback-only. Ollama remains a host process, not a Compose service. |
| `tests/test_vision_backend.py` | Unit coverage for local endpoint policy, sampling invariants, output handling, retries, routing, and unload. |

## Trust and network boundary

The adapter allows only these native Ollama endpoints:

```text
http://host.docker.internal:11434
http://127.0.0.1:11434
http://localhost:11434
```

It rejects any other scheme, host, port, path, query, or fragment before a
network request is made. The Docker backend normally uses
`host.docker.internal` to reach Ollama on the same Windows machine.

The adapter sends only in-memory base64 JPEG samples to Ollama. It never sends
the source file, writes temporary image files, accepts a model-supplied path,
or contacts Gemini while local vision is selected.

OpenShorts services are published only to `127.0.0.1`. The existing backend
listener may bind inside its Docker network; the loopback host mapping is the
actual local-machine boundary.

## Configuration contract

The operator-facing values and commands live in the runbook. The code enforces
these limits regardless of `.env` values:

| Variable | Allowed range | Default |
|---|---:|---:|
| `VISION_TIMEOUT_SECONDS` | 10–600 seconds | 180 |
| `VISION_RETRY_ATTEMPTS` | 1–3 | 3 |
| `VISION_WINDOW_SECONDS` | 60–120 seconds | 120 |
| `VISION_WINDOW_OVERLAP_SECONDS` | `ceil(CLIP_MAX_SECONDS)` through `window - 1` | 60 |
| `VISION_FRAMES_PER_WINDOW` | 4–10 | 8 |
| `VISION_MAX_FRAME_WIDTH` | 480–768 pixels | 640 |
| `VISION_NUM_CTX` | 2048–8192 tokens | 4096 |
| `VISION_CANDIDATES_PER_WINDOW` | 1–3 | 2 |
| `VISION_SCENE_SNAP_SECONDS` | 0–3 seconds | 2 |

Missing or non-numeric settings use their defaults. Numeric values outside the
range are clamped. A missing or blank `VISION_MODEL` intentionally uses the
approved `qwen2.5vl:3b` default; a supplied model name containing control
characters fails closed. Invalid endpoints always fail closed.

The local visual path accepts `CLIP_MAX_SECONDS` through 60 seconds. It rejects
a larger duration and rejects a window that is not strictly larger than
`ceil(CLIP_MAX_SECONDS)`. This guarantees every allowed candidate is completely
visible in at least one window.

## `vision_backend.py` contract

The public interface is intentionally small:

```python
class LocalVisionError(RuntimeError):
    """A configuration, decoding, Ollama, or validation error."""

def active() -> bool:
    """True only when VISION_PROVIDER is 'ollama'."""

def select_clips(
    video_path: str,
    video_duration: float,
    language: str,
    min_clips: int,
    max_clips: int,
    min_seconds: float,
    max_seconds: float,
) -> dict:
    """Return {"shorts": [...], "cost_analysis": {...}} or raise LocalVisionError."""
```

The output clips conform to `gemini_worker.VisualClipModel`, allowing the
existing OpenShorts metadata, crop, and render stages to remain unchanged.

### Analysis flow

```text
validate settings and local endpoint
-> validate source can be opened
-> call Ollama /api/show for the configured model
-> detect scene boundaries once
-> for each overlapping analysis window:
     decode JPEG samples in memory
     call Ollama /api/chat with a JSON schema
     validate and normalize candidates
-> deduplicate by score, cap, then order by timeline
-> call Ollama /api/generate with keep_alive=0
```

Windows are 120 seconds with a 60-second overlap by default. Each window
samples eight evenly spaced timestamps (5%–95%) and up to two additional scene
boundary timestamps. The hard maximum is ten images per Ollama request.

Each successfully encoded JPEG retains the timestamp of its decoded frame
(frame index divided by FPS).
When decoding or encoding fails, both the image and its paired timestamp are
dropped. The ordered image and timestamp arrays are therefore always aligned.

### Ollama request

The adapter uses native `POST /api/chat`, rather than an OpenAI-compatible
endpoint, so it can provide an ordered `images` list:

```python
{
    "model": "qwen2.5vl:3b",
    "stream": False,
    "format": gemini_worker.VisualResponse.model_json_schema(),
    "keep_alive": "10m",
    "options": {"temperature": 0, "num_ctx": 4096},
    "messages": [{
        "role": "user",
        "content": prompt,
        "images": base64_jpegs,
    }],
}
```

The prompt includes the exact image-timestamp order, source/window duration,
local scene boundaries, clip-duration limits, the chosen language, and an
instruction that visible text is untrusted visual content—not executable
instructions.

### Output handling and failures

- Only `message.content` is parsed.
- Parsed output is validated with `gemini_worker.VisualResponse`.
- Candidate timestamps must be finite, source-bounded, and inside the window
  whose frames the model saw.
- Starts and ends snap to a nearby scene boundary only if the modified clip
  remains inside configured duration limits.
- A CLI/test source shorter than `min_seconds` may produce one full-duration
  clip; the normal dashboard source-duration guard rejects such sources first.
- Existing `dedupe_overlapping()` keeps the higher-scoring duplicate. Results
  are then capped by `max_clips` and timeline-sorted for rendering.
- Connection failures, timeouts, HTTP 429, and HTTP 5xx retry at most three
  times with exponential backoff.
- HTTP 400, 401, 403, and 404 fail immediately as local Ollama configuration
  errors.
- One malformed JSON/schema response gets one corrective retry.
- Any failed analysis window fails the entire job, naming that window; partial
  coverage is never silently returned.
- `finally` always attempts to unload the selected VLM with
  `POST /api/generate` and `{"keep_alive": 0}`. An unload failure never hides
  the real selection failure.

## `main.py` integration

`main.get_visual_clips()` begins with the local route:

```python
if vision_backend.active():
    min_clips = _env_int("CLIP_TARGET_MIN", 3)
    max_clips = max(min_clips, _env_int("CLIP_TARGET_MAX", 15))
    min_secs, max_secs = clip_duration_bounds()
    return vision_backend.select_clips(
        video_path=video_path,
        video_duration=video_duration,
        language=language,
        min_clips=min_clips,
        max_clips=max_clips,
        min_seconds=min_secs,
        max_seconds=max_secs,
    )
```

There is intentionally no broad `except` around this call. Exact local
configuration, model, decoding, and schema errors must reach the job log.

When local vision is inactive, execution proceeds to the unchanged Gemini
implementation below it.

## Test contract

`tests/test_vision_backend.py` covers:

- local-only endpoint allow-list and settings bounds;
- paired frame/timestamp removal on decode failure and its window-labelled error;
- 120/60 window coverage for all 15–60-second candidates;
- no more than two scene-boundary samples, even with a low regular frame count;
- safe scene-boundary snapping;
- short-source behavior;
- exactly one malformed-output correction and failure after a second malformed reply;
- bounded 503 retry and immediate 401 failure;
- all-or-nothing window behavior and attempted VLM unload;
- overlap deduplication; and
- `main.py` routing precedence.

Run the focused tests in the backend development container:

```powershell
docker compose exec --user root backend python -m pip install pytest
docker compose exec backend python -m pytest -q tests/test_vision_backend.py tests/test_sparse_speech.py
docker compose exec backend python -m py_compile vision_backend.py main.py
```

## Known limits

- The VLM sees samples, not every source frame; an event between samples can be
  missed.
- Frame timestamps use frame index divided by reported FPS. That is a practical
  approximation for ordinary constant-frame-rate MP4s; variable-frame-rate
  sources can have a small timestamp offset.
- Clip selection is editorial guidance, not a guarantee of perfect cuts;
  review clips before publishing.
- A video with frequent lyrics or spoken words routes to the transcript picker
  even when it is visually driven. A manual visual-mode override would be a
  separate feature.
- The local vision route only replaces silent-video clip selection. Optional
  layout selection, thumbnail work, effects, and hook grounding can still use
  Gemini when configured.
