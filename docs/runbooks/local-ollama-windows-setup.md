# Runbook: Run OpenShorts with local Ollama on Windows

This is the complete operator setup for the private local clip-selection
workflow:

```text
Talking video
  -> Whisper transcript (local)
  -> qwen2.5:7b through Ollama (local)
  -> selected clip timestamps
  -> existing OpenShorts render pipeline

No-audio or sparse-speech video
  -> audio-stream check, then Whisper only when an audio track exists
  -> timestamped JPEG samples (local)
  -> qwen2.5vl:3b through Ollama (local)
  -> selected clip timestamps
  -> existing OpenShorts render pipeline
```

No source video or clip-selection frame sample is sent to Gemini when local
visual selection is enabled. Set `GEMINI_CLIP_ANALYSIS_ENABLED=0` below to
also prevent optional Gemini layout, on-screen, and hook helpers from making
frame requests. That setting is independent of the explicit per-clip AI
thumbnail action.

For the code architecture and test contract, see
[`../architecture/local-ollama-vision.md`](../architecture/local-ollama-vision.md).

## 1. Prerequisites

You need:

- An OpenShorts branch containing the local Ollama vision implementation:
  `feature/local-ollama-vision` for local vision alone, or
  `feature/openshorts-postiz-youtube` for local vision plus direct local
  Postiz → YouTube handoff.
- Docker Desktop running in Linux-container mode.
- NVIDIA drivers and Docker GPU support working for OpenShorts.
- Ollama installed on the Windows host, not inside Docker.
- At least 12 GB of GPU VRAM and 32 GB RAM for the selected configuration.

Confirm the intended branch before editing any configuration:

```powershell
git branch --show-current
git log -1 --oneline
```

Either of these branches is valid:

```text
feature/local-ollama-vision
feature/openshorts-postiz-youtube
```

### Required Docker GPU override

Create `docker-compose.override.yml` beside `docker-compose.yml`, or merge this
`backend` block into an existing override. This exposes the NVIDIA GPU to the
OpenShorts backend container and installs its GPU dependencies during build:

```yaml
services:
  backend:
    build:
      context: .
      args:
        GPU: "1"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, video]
```

`video` is required for NVENC encoding. Confirm GPU access before continuing:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

## 2. Install and configure Ollama

Install Ollama on Windows if it is not already installed. Then open a fresh
PowerShell window and download both models:

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:3b
ollama list
```

Expected model names include:

```text
qwen2.5:7b
qwen2.5vl:3b
```

Configure the Windows Ollama process to keep only one model and one request in
GPU memory at a time:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "8192", "User")
```

These are **Windows user environment variables**, not entries in OpenShorts'
`.env`. Quit Ollama completely from the system tray and start it again after
setting them.

Confirm that Ollama is serving locally:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

This should return JSON listing the downloaded models.

## 3. Configure OpenShorts `.env`

At the OpenShorts repository root, edit the existing `.env` file. Do not
overwrite an existing file containing keys or other personal settings.

Add or replace the local-model settings with this block:

```env
# Talking videos: Whisper transcript -> local text model selects clip moments.
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=
LLM_SCORE_BATCH=3
LLM_TIMEOUT=600

# Silent / wordless videos: timestamped frame samples -> local vision model.
VISION_PROVIDER=ollama
VISION_BASE_URL=http://host.docker.internal:11434
VISION_MODEL=qwen2.5vl:3b
VISION_TIMEOUT_SECONDS=300
VISION_RETRY_ATTEMPTS=3

# Local visual-analysis windows. The maximum supported local clip length is 60s.
# Sources are limited to 120 minutes and 120 analysis windows. Overlap must
# leave at least a 15-second step, so do not raise it above 105 here.
VISION_WINDOW_SECONDS=120
VISION_WINDOW_OVERLAP_SECONDS=60
# Conservative qwen2.5vl:3b profile for a 12 GB laptop GPU.
# Scene sampling may add up to two extra images to this request.
VISION_FRAMES_PER_WINDOW=4
# Compatibility name: this is the JPEG's maximum longest edge, including for
# portrait video; it is not a width-only cap.
VISION_MAX_FRAME_WIDTH=480
VISION_NUM_CTX=8192
VISION_CANDIDATES_PER_WINDOW=2
VISION_SCENE_SNAP_SECONDS=2

# A 12 GB laptop GPU should process one OpenShorts job at a time.
MAX_CONCURRENT_JOBS=1
ASR_GPU_CONCURRENCY=1

# Keep Gemini out of every automatic clip-analysis/helper step. This does not
# disable the separate, user-triggered per-clip AI thumbnail action.
GEMINI_CLIP_ANALYSIS_ENABLED=0

# Optional: enable only if you want the per-clip "generate AI thumb" button.
# Gemini receives one locally extracted JPG reference frame, not the MP4.
# Nano Banana 2 has no Gemini API free tier: this key must belong to a
# Gemini API project with billing enabled and image-generation quota.
CLIP_AI_THUMBNAILS_ENABLED=1
GEMINI_API_KEY=your_google_ai_studio_key
CLIP_THUMBNAIL_IMAGE_MODEL=gemini-3.1-flash-image
CLIP_THUMBNAIL_IMAGE_SIZE=1K
CLIP_THUMBNAIL_MAX_CONCURRENT_GENERATIONS=1
```

Use plain values, one per line. Do not paste smart quotes or hidden characters.
Standard dotenv parsers support valid quoted values, but unquoted values avoid
copy/paste and Compose interpolation mistakes.

Important limits:

- Keep `CLIP_MAX_SECONDS` at `60` or lower while `VISION_PROVIDER=ollama` is
  enabled. A larger value fails clearly rather than producing incompletely
  sampled clips.
- Keep `VISION_WINDOW_SECONDS` between
  `max(60, ceil(CLIP_MAX_SECONDS) + 15)` and `120`, and choose overlap between
  `ceil(CLIP_MAX_SECONDS)` and
  `VISION_WINDOW_SECONDS - 15`. Invalid window geometry is rejected instead
  of silently changed.
- Local vision is deliberately limited to source videos no longer than 120
  minutes and no more than 120 analysis windows. Split longer silent footage
  before processing.
- Do not add an Ollama service or port mapping to `docker-compose.yml`.
  `host.docker.internal` lets the backend container reach Ollama on the same
  Windows laptop.
- OpenShorts' own dashboard, API, and renderer are bound to `127.0.0.1`; other
  devices on Wi-Fi cannot reach them.
- `gemini-3.1-flash-image` (Nano Banana 2) cannot generate images using a
  Gemini API free-tier key. Enable billing on the Google Cloud project that
  owns `GEMINI_API_KEY`; a Google AI Pro chat subscription is separate from
  Gemini API billing. This affects only the optional **generate AI thumb**
  action, never local clip generation or Postiz upload.

## 4. Build and start OpenShorts

From the OpenShorts repository root:

```powershell
docker compose config
docker compose down
docker compose up --build -d
docker compose ps
```

The first build can take several minutes. `docker compose ps` should show
`openshorts-backend`, `openshorts-frontend`, and `openshorts-renderer` running.

Open the dashboard:

```text
http://localhost:5175
```

For later starts, when nothing changed:

```powershell
docker compose up -d
```

## 5. Verify container-to-Ollama connectivity

The following command must print `200`:

```powershell
docker exec openshorts-backend python -c "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags', timeout=10).status_code)"
```

If it does not, do not start a video job yet. Check that Ollama is running on
Windows and repeat the host-side test from step 2.

## 6. Generate test clips

Use Clip Generator in the dashboard and test these three sources:

| Source | Expected route | Expected result |
|---|---|---|
| Video with spoken dialogue | Whisper → `qwen2.5:7b` | Speech-driven clips with subtitles. |
| Video with no audio stream | Qwen2.5-VL directly | Visual clips, no subtitles. |
| Music/ambience-only video | Whisper establishes sparse speech → Qwen2.5-VL | Visual clips, no subtitles. |

Watch backend logs during the job:

```powershell
docker compose logs -f backend
```

For a silent source, look for:

```text
Silent video — analyzing with local Ollama vision
```

For a talking source, look for the normal local text-model analysis log. It
should not load the vision model.

## 7. Confirm GPU use and memory release

During a silent-video job, in a second PowerShell window:

```powershell
nvidia-smi -l 1
```

Expected sequence:

1. Ollama / `qwen2.5vl:3b` occupies GPU memory during visual selection.
2. OpenShorts asks Ollama to unload the vision model once selection completes.
3. OpenShorts renders the selected clips.

You can also inspect loaded Ollama models:

```powershell
ollama ps
```

If an interrupted job leaves the vision model in memory, release it manually:

```powershell
ollama stop qwen2.5vl:3b
```

## 8. Everyday operations and recovery

| Need | Command |
|---|---|
| Start OpenShorts | `docker compose up -d` |
| Follow backend logs | `docker compose logs -f backend` |
| Stop containers cleanly | `docker compose down` |
| Force-stop stuck containers | `docker compose kill` then `docker compose down` |
| Rebuild after code/config change | `docker compose up --build -d` |
| Check container state | `docker compose ps` |
| Release a loaded VLM | `ollama stop qwen2.5vl:3b` |

Use `docker system prune` or `docker builder prune` only when you intentionally
want to remove unused Docker data. They are not part of ordinary startup or
shutdown.

## 9. Troubleshooting

| Symptom | Check / resolution |
|---|---|
| `Connection refused` or timeout to `host.docker.internal:11434` | Start Ollama, run the host-side API test, then run the container connectivity test from step 5. |
| `HTTP 404` from `/api/show` | Run `ollama pull qwen2.5vl:3b`; the vision model is missing. |
| `CLIP_MAX_SECONDS up to 60` error | Set `CLIP_MAX_SECONDS=60` or remove it if it was overridden. |
| GPU out-of-memory | Keep `MAX_CONCURRENT_JOBS=1`, `ASR_GPU_CONCURRENCY=1`, `OLLAMA_NUM_PARALLEL=1`, and `OLLAMA_MAX_LOADED_MODELS=1`. Close other GPU-heavy apps. |
| `request ... exceeds the available context size` | Use the conservative profile above: 4 frames, 480 px maximum JPEG longest edge, and `VISION_NUM_CTX=8192`; recreate the OpenShorts backend after changing `.env`. |
| Silent job fails after retry attempts | This is an Ollama transport/server/model/context problem and fails the job deliberately. Check Ollama is running, `ollama list` includes `qwen2.5vl:3b`, and keep the conservative image/context settings. |
| `Local Ollama vision returned no usable clip candidates` | OpenShorts skipped any individual frame-decode or twice-malformed response window and tried the remaining windows. The final error includes up to three named diagnostics. It means every window was skipped, empty, or produced no valid 15–60 second visual candidate. |
| `VISION_WINDOW_SECONDS` / `VISION_WINDOW_OVERLAP_SECONDS` configuration error | Restore the 120/60 defaults. The settings must leave a 15-second step and preserve a full maximum-length clip in one window. |
| Source-video / analysis-window limit error | Split a silent source longer than 120 minutes, or lower a deliberately high overlap so no more than 120 windows are required. |
| Docker daemon error | Start Docker Desktop, wait until it reports that the engine is running, then rerun `docker compose up -d`. |
| Changes in `.env` appear ignored | Recreate the backend: `docker compose up -d --force-recreate backend`. |

## 10. Switch back to Gemini

No code rollback is needed.

To switch **silent-video selection only** back to Gemini, remove
`VISION_PROVIDER=ollama` and set:

```env
GEMINI_CLIP_ANALYSIS_ENABLED=1
```

To also switch **talking-video transcript selection** back to Gemini, use:

```env
LLM_PROVIDER=gemini
GEMINI_CLIP_ANALYSIS_ENABLED=1
```

Leave `CLIP_AI_THUMBNAILS_ENABLED=1` unchanged if you still want the
on-demand Nano Banana thumbnail button. Ensure `GEMINI_API_KEY` is set in the
backend `.env` (or provide the existing dashboard BYOK key), then recreate
the backend:

```powershell
docker compose up -d --force-recreate backend
```

You may also mix the routes: local Qwen text selection with Gemini vision, or
Gemini text selection with local Qwen-VL vision.
