# OpenShorts Pipeline Checkpointing — Implementation Map

## Status

This is the file-by-file implementation companion to:

- [Pipeline Checkpointing — Problem Statement](pipeline-checkpointing-problem-statement.md)
- [Pipeline Checkpointing — Design](pipeline-checkpointing-design.md)
- [Pipeline Checkpointing — Actual Code Flow](pipeline-checkpointing-code-flow.md)

The design defines required behaviour and safety rules. This document defines
where those rules enter the current OpenShorts codebase.

The implementation now follows this map on the checkpointing feature branch.
The dependency order below remains the review and backport order: stable job
identity, atomic state and exclusive ownership must precede reusable provider
or render outputs.

Symbol names, rather than line numbers, are the authoritative anchors. Line
numbers move as the repository changes.

## Scope

The initial implementation adds:

- `PIPELINE_CHECKPOINTS_ENABLED=true|false`, defaulting to `false`;
- persistent project identity and immutable resolved job configuration;
- crash-safe job and resume state;
- single-writer ownership of a project directory;
- source, ASR, source-analysis, local text-model, local vision-model,
  clip-plan and render checkpoints;
- restart-safe partial completion;
- cleanup, archive and download exclusions for sensitive state;
- a standalone CLI contract using `--project-dir`;
- Windows Docker restart and process-kill coverage.

It does not change model prompts, clip-selection quality, render styles or the
public response format except where a `partial` terminal result must report
missing planned clips.

## New Modules

### `job_state.py`

Create this at the repository root beside `app.py` and `main.py`.

It owns durable job-level state, not model-stage checkpoint payloads.

Required public surface:

```python
class JobStateError(RuntimeError):
    pass


class JobLeaseBusy(JobStateError):
    pass


class StateFileMissing(JobStateError):
    pass


class StateFileInvalid(JobStateError):
    pass


def atomic_write_bytes(path: str, payload: bytes, mode: int = 0o600) -> None:
    ...


def atomic_write_json(path: str, payload: dict, mode: int = 0o600) -> None:
    ...


def read_json(path: str) -> dict | None:
    ...


def read_required_json(path: str, validator) -> dict:
    """Raise StateFileMissing or StateFileInvalid; never collapse both to None."""
    ...


def stable_job_paths(project_dir: str, job_id: str) -> dict:
    ...


def write_job_config(project_dir: str, config: dict) -> None:
    ...


def load_job_config(project_dir: str) -> dict:
    """Raise StateFileMissing or StateFileInvalid when not usable."""
    ...


def write_job_state(project_dir: str, state: dict) -> None:
    ...


def load_job_state(project_dir: str) -> dict:
    """Raise StateFileMissing or StateFileInvalid when not usable."""
    ...


def write_resume_manifest(project_dir: str, manifest: dict) -> None:
    ...


def load_resume_manifest(project_dir: str) -> dict | None:
    ...


def load_resume_secret_keyring(env: dict | None = None) -> dict:
    ...


def write_resume_secrets(
    project_dir: str,
    job_id: str,
    configuration_revision: int,
    purpose: str,
    secrets: dict,
    keyring: dict,
) -> None:
    ...


def load_resume_secrets(
    project_dir: str,
    job_id: str,
    configuration_revision: int,
    purpose: str,
    keyring: dict,
) -> dict | None:
    ...


def commit_terminal_state(
    project_dir: str,
    expected_execution_generation: int,
    terminal_status: str,
    terminal_artifacts: dict,
    missing_clip_indices: list[int],
    action_context: dict,
) -> dict:
    """Allocate revision and atomically persist decision, artifacts, outbox."""
    ...


def transition_terminal_action(
    project_dir: str,
    action: str,
    expected_state: str,
    new_state: str,
    details: dict | None = None,
) -> dict:
    ...


def transition_terminal_action_item(
    project_dir: str,
    action: str,
    item_id: str,
    expected_state: str,
    new_state: str,
    details: dict | None = None,
) -> dict:
    """Atomically update one nested item and recompute its parent action."""
    ...


def required_terminal_actions(
    terminal_status: str,
    action_context: dict,
) -> dict:
    """Derive and validate the complete initial outbox; callers cannot choose it."""
    ...


class JobLease:
    @classmethod
    def acquire(
        cls,
        project_dir: str,
        job_id: str,
        *,
        blocking: bool = False,
    ):
        ...

    def release(self) -> None:
        ...
```

Implementation rules:

- write a sibling temporary file, flush it, `fsync` it, atomically replace the
  target, then `fsync` the parent directory;
- use an exclusive OS-level lock, not a check-then-create existence test;
- keep the lock handle open for the complete execution;
- use lowercase fixed filenames from the architecture design;
- reject project paths that resolve outside the configured output root when
  called by `app.py`;
- do not log state payloads because they may contain source metadata;
- never write API keys, bearer tokens, cookies or webhook secrets to plaintext
  JSON;
- make the helper work on Windows and POSIX using a small adapter for
  `msvcrt.locking` and `fcntl.flock`, with tests for both branches;
- move the generic atomic-write logic currently used by
  `app.py::_write_postiz_state` into this shared module rather than maintaining
  two subtly different implementations.

Critical job state never uses a `dict | None` loader when the distinction
matters. Missing state can mean a legacy/new job; corrupt or schema-invalid
state is an explicit recovery error and must not be silently recreated over.

Stable job-level files:

```text
.job-config.json
.job-state.json
.resume.json
.resume-secrets.enc
.run.lock
```

`stable_job_paths()` also defines:

```text
source.mp4
<job_id>_metadata.json
.checkpoints/
```

The retained source extension may be preserved when required, but its name
must be stable and stored in `.job-config.json`; it must not depend on the
YouTube title or client filename.

### `checkpoint_store.py`

Create this at the repository root.

It owns compute-stage checkpoint envelopes and compatibility validation.

Required public surface:

```python
class CheckpointError(RuntimeError):
    pass


def pipeline_checkpoints_enabled(env: dict | None = None) -> bool:
    ...


def canonical_json_bytes(value) -> bytes:
    ...


def sha256_bytes(value: bytes) -> str:
    ...


def sha256_file(path: str) -> str:
    ...


def compatibility_key(stage: str, inputs: dict) -> str:
    ...


def expected_source_request_key(job_config: dict) -> str:
    ...


def expected_asr_key(
    job_config: dict,
    source_identity: dict,
    effective_route: str,
) -> str:
    ...


def validate_asr_route(value) -> str:
    ...


def valid_asr_payload(payload, source_duration: float | None = None) -> bool:
    ...


def completed_selection_boundary(
    config: dict,
    *,
    current_enablement: bool,
) -> str | None:
    ...


def expected_clip_plan_key(
    job_config: dict,
    source_identity: dict,
    route_context: dict,
) -> str:
    ...


def expected_render_key(
    job_config: dict,
    source_identity: dict,
    clip_plan_envelope: dict,
    clip: dict,
    source_analysis: dict,
) -> str:
    ...


def build_source_identity(path: str, probe: dict) -> dict:
    ...


class CheckpointStore:
    def __init__(
        self,
        project_dir: str,
        job_config: dict,
        enabled: bool,
        execution_generation: int,
        source_identity: dict | None = None,
    ):
        ...

    def load_source(self, expected_request_key: str) -> dict | None:
        ...

    def save_source(
        self,
        expected_request_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        ...

    def bind_source(self, source_identity: dict) -> None:
        ...

    def load_fixed(self, stage: str, expected_key: str) -> dict | None:
        ...

    def load_routed_fixed(
        self,
        stage: str,
        validate_route_context,
        expected_key_for_route,
    ) -> dict | None:
        """Validate stored bounded route context, recalculate key, validate."""
        ...

    def save_fixed(
        self,
        stage: str,
        expected_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        ...

    def load_keyed(
        self,
        stage: str,
        expected_key: str,
    ) -> dict | None:
        ...

    def save_keyed(
        self,
        stage: str,
        expected_key: str,
        payload: dict,
        provenance: dict,
    ) -> bool:
        ...

    def validate_artifact(self, descriptor: dict) -> bool:
        ...

    @classmethod
    def open_existing_project(
        cls,
        project_dir: str,
        *,
        current_enablement: bool,
    ):
        """Load required state and bind a validated retained source."""
        ...

    def load_current_clip_plan(self, expected_key_for_route) -> dict | None:
        ...

    def valid_render_artifacts(self, clip_plan_envelope: dict) -> dict[int, dict]:
        """Recalculate current render keys and return validated artifacts."""
        ...
```

Implementation rules:

- `pipeline_checkpoints_enabled()` accepts only case-insensitive `true` or
  `false`; unset means `false`, and every other value raises a clear
  configuration error;
- when disabled, the store exposes the same methods, returns cache misses and
  writes nothing;
- fixed and keyed paths are derived internally from allowlisted stage names;
  callers do not pass arbitrary relative paths;
- keyed checkpoint filenames are the `expected_key` itself. There is no second
  `unit_key`; the canonical unit/window identity is one of the hashed
  compatibility inputs;
- unit filenames are lowercase SHA-256 values, never transcript text, URLs,
  model names or source titles;
- readers validate schema, implementation version, job ID, source identity,
  configuration revision, compatibility key and payload hash;
- incompatible or corrupt files are cache misses, not fatal job errors;
- save failure logs a concise warning, returns `False`, and does not fail
  successful compute;
- checkpoint payloads contain JSON and artifact descriptors only. They do not
  copy videos, frames or render output into `.checkpoints/`;
- every write uses `job_state.atomic_write_json()`.

The store begins unbound so it can validate the retained `source.json` using
the source-request/cap compatibility key. After that artifact is reused or
newly ingested, call `bind_source()` exactly once. Every non-source checkpoint
operation fails closed until a source identity is bound.

`load_routed_fixed()` is required for `asr.json` and `clip-plan.json`. It reads
only the allowlisted fixed path, validates the route context through a
stage-owned validator, invokes the pure expected-key callback with that
validated context, and then performs full envelope validation. The stored
compatibility key is never trusted as the expected value.

For ASR, route context is a single bounded effective-backend enum. For a clip
plan it is:

```json
{
  "selection_route": "speech-local",
  "asr_effective_route": "parakeet-to-whisper-cpu",
  "asr_outcome": "speech"
}
```

Silent examples use `selection_route: silent-local|silent-gemini` and
`asr_outcome: no_audio|sparse_speech`. Every field is a bounded enum. This is
enough to recalculate the transitive ASR and selection compatibility keys
without reproducing a transcript or model response.

`open_existing_project()` is the app-side entry point used by retry cleanup and
terminal recovery. It loads required job files with missing-versus-invalid
distinction, validates and binds `source.json`, and applies current deployment
enablement. `valid_render_artifacts()` uses the same render-key builder as
`main.py`; app code does not reimplement render compatibility.

The module-level expected-key builders, ASR validator and completed-selection
boundary resolver are shared by `main.py` and `app.py`. Provider modules supply
resolved descriptors and validated payloads, but neither process maintains a
private copy of compatibility or recovery-route logic.

### Encrypted resume-secret configuration

Add an authenticated-encryption dependency to `requirements.txt` and use
AES-256-GCM from that maintained library. Do not implement cryptography from
Python primitives.

The keyring is server configuration, not project configuration:

```dotenv
# Required only when a request-scoped secret, such as a webhook HMAC secret,
# must survive a restart.
RESUME_SECRET_ACTIVE_KEY_ID=v1
RESUME_SECRET_KEYS=v1:<base64url-encoded-32-byte-key>
```

During rotation, `RESUME_SECRET_KEYS` may contain comma-separated old keys:

```dotenv
RESUME_SECRET_KEYS=v2:<new-key>,v1:<old-key>
```

New writes use the active key. Reads select by the envelope `key_id`. The AEAD
additional authenticated data is canonical JSON containing `job_id`,
`configuration_revision`, `purpose` and `format_version`. If a required key is
missing, the job enters `waiting_for_credentials`; it never sends an unsigned
callback or substitutes another secret.

## Resolved Configuration Contract

### `app.py`

At request acceptance, resolve every request-specific and environment-specific
processing option once. Some values such as plan watermarking and source caps
are known only after reservation and quality-gate work. Persist the final
resolved object to `.job-config.json` after those decisions but before the job
can be enqueued.

The resolved object includes at least:

```json
{
  "schema_version": 1,
  "job_id": "<uuid>",
  "configuration_revision": 1,
  "created_at": "<UTC ISO-8601>",
  "source": {
    "kind": "url|upload|upload_slot|thumbnail_session",
    "original_name": "<display only>",
    "retained_relative_path": "source.mp4",
    "url_identity": "<redacted canonical identity or null>",
    "cap_minutes": null
  },
  "pipeline": {
    "output_format": "auto",
    "layouts": [],
    "auto_hook": false,
    "auto_hook_style": null,
    "captions": true,
    "target_clips": null,
    "clip_min_seconds": null,
    "clip_max_seconds": null,
    "watermark": false
  },
  "asr": {},
  "scene_detection": {},
  "text_selection": {},
  "vision_selection": {},
  "render": {},
  "checkpointing_enabled": true
}
```

Record resolved values, not merely environment-variable names. Credential
fields are prohibited.

Authoritative processing fields and their current sources:

| Job-config field | Current request/environment source |
|---|---|
| `source.kind`, `source.provider_identity`, `source.retained_relative_path` | `file`, `url`, `upload_id`, or `thumbnail_session_id`; sanitized by `process_endpoint` |
| `source.max_source_minutes`, `source.source_cap_minutes` | `max_minutes`, `MAX_SOURCE_MINUTES`, `SOURCE_CAP_MINUTES` |
| `source.download_skip_statics`, `source.downloader_policy` | `DOWNLOAD_SKIP_STATICS`, current yt-dlp/download route policy |
| `pipeline.output_format` | `output_format` / `--format` |
| `pipeline.layouts` and layout samples | `layouts`, `AUTO_LAYOUT`, `LAYOUT_SAMPLE_FRAMES`, `LAYOUT_SAMPLE_WIDTH` |
| `pipeline.layout_provider` | sanitized provider endpoint, `GEMINI_MODEL`/resolved revision, layout prompt/schema versions, enabled layout feature set |
| `pipeline.burned_subtitles` | `SUBTITLE_BAND_CHECK`, sanitized provider/model revision, fixed sample/crop/threshold constants and detector implementation version |
| `pipeline.auto_hook`, `auto_hook_style`, `auto_hook_seconds` | request fields plus `AUTO_HOOK`, `AUTO_HOOK_STYLE`, `AUTO_HOOK_SECONDS` |
| `pipeline.captions` | request field plus `AUTO_CAPTIONS` |
| `pipeline.watermark` | resolved user plan plus `WATERMARK` |
| `selection.target_min`, `target_max` | `target_clips`, `CLIP_TARGET_MIN`, `CLIP_TARGET_MAX`, or duration-aware defaults |
| `selection.clip_min_seconds`, `clip_max_seconds` | request fields, `CLIP_MIN_SECONDS`, `CLIP_MAX_SECONDS` |
| `selection.score_batch`, `detail_fallback_windows` | `LLM_SCORE_BATCH`, `LLM_DETAIL_FALLBACK_WINDOWS` |
| `asr.requested_backend`, `fallback_policy` | `TRANSCRIBE_BACKEND` and the code-defined fallback policy |
| `asr.whisper` | `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE`, `WHISPER_TRANSCRIBE_PARAMS` |
| `asr.parakeet` | `PARAKEET_MODEL_ID`, `PARAKEET_VAD_BATCH`, VAD configuration |
| `asr.sparse_policy` | `MIN_SPEECH_WORDS`, `MIN_SPEECH_WORDS_PER_MIN` |
| `scene_detection` | `SCENE_ENGINE`, `TRANSNETV2_DEVICE`, `TRANSNETV2_THRESHOLD`, `SCENE_MIN_SEC` |
| `text_selection.provider` | `LLM_PROVIDER`, sanitized `LLM_BASE_URL`, `LLM_MODEL`, resolved immutable digest |
| `vision_selection` | `VISION_PROVIDER`, sanitized `VISION_BASE_URL`, `VISION_MODEL`, digest, window/overlap/frame/width/context/candidate/snap settings |
| `render.encoder` | `FFMPEG_ENCODER`, `FFMPEG_GPU_CUT`, `CUT_NVENC_PRESET`, `CUT_NVENC_CQ`, `AUDIO_NORMALIZE` |
| `render.reframe` | `REFRAME_ENGINE`, `GENERAL_CONTENT_HEIGHT_RATIO`, `SPARSE_TRAJECTORY`, detector/model implementation IDs |
| `render.tracking` | `YOLO_MODEL_PATH` content digest, `DETECT_STRIDE`, `JUMP_CONFIRM_FRAMES`, `SCENE_CUT_RESET`, MediaPipe/face/person detector versions and thresholds |
| `render.overlays` | hook/caption style defaults, prompt/grounding versions, font and watermark asset digests |
| `render.parallelism` | `CLIP_WORKERS`, `CLIP_RETRY_PAUSE_SECONDS` |

Timeouts, retries and concurrency are persisted for reproducible recovery but
are compatibility inputs only when they can alter a completed semantic result.
Secrets such as `GEMINI_API_KEY`, `LLM_API_KEY`, cookies and proxy credentials
are represented only as required capabilities.

Deployment enablement is a kill switch:

```text
effective_checkpointing =
    current PIPELINE_CHECKPOINTS_ENABLED
    AND job_config.checkpointing_enabled
```

Thus changing the current deployment flag to `false` immediately disables
compute checkpoint reads/writes for resumed jobs. Changing it from `false` to
`true` does not retroactively checkpoint a job accepted with the feature off;
that requires a deliberate new configuration revision.

### `main.py`

Add internal arguments:

```text
--project-dir <persistent job directory>
--job-id <job UUID>
--managed-job
--execution-generation <integer>
```

Rules:

- checkpoint-enabled standalone CLI use requires `--project-dir`;
- standalone CLI creates or loads project identity and acquires `JobLease`;
- `app.py` passes `--managed-job`, owns the lease around the subprocess and
  passes the current execution generation;
- `main.py` validates that `--project-dir`, `--job-id` and loaded
  `.job-config.json` agree;
- `--output` remains the delivery directory for compatibility, but managed
  runs set it to the same path as `--project-dir`;
- a checkpoint-disabled CLI run retains current behaviour and does not require
  a project directory.

## Exact Changes by Existing File

### 1. `app.py`

#### Import and environment initialization

Current anchor: the module import block and `load_dotenv()`.

Change:

1. import and call `load_dotenv()` before modules that read model or layout
   environment values;
2. import `job_state` after standard-library imports;
3. keep provider imports after dotenv, or remove their import-time environment
   constants in the provider changes below.

This prevents saved resolved configuration from disagreeing with values
captured during module import.

#### Atomic state helper consolidation

Current anchors:

- `_fsync_parent_directory`
- `_write_postiz_state`

Change:

- move generic atomic byte/JSON writing to `job_state.py`;
- update the Postiz state writer to call the shared helper;
- do not alter the Postiz state schema as part of checkpointing.

#### Request acceptance and job creation

Current anchor: `process_endpoint`.

Insert changes in this order:

1. parse and validate existing request fields;
2. create `job_id` and `job_output_dir`;
3. copy, move or hardlink upload sources into the job directory under the
   stable retained name;
4. run current reservation, quality-gate, source-cap and plan decisions;
5. resolve model, ASR, scene, layout, selection and render configuration,
   including the final watermark/cap values;
6. build immutable `.job-config.json`;
7. create `.job-state.json` with status `accepted`,
   `configuration_revision=1` and `execution_generation=0`;
8. write `.owner` atomically as a backward-compatible projection of the owner
   stored authoritatively in `.job-state.json`;
9. write sanitized `.resume.json`;
10. encrypt separately any resumability secret that genuinely must survive;
11. only after all required durable files succeed, create the in-memory job
    and enqueue it.

Source changes:

- direct uploads currently write to `UPLOAD_DIR`; write the final owned copy to
  `job_output_dir`;
- `upload_slot` and `thumbnail_session` sources are likewise moved, hardlinked
  or copied into the job directory;
- the job directory becomes the single cleanup boundary;
- URL jobs keep the stable source produced by `main.py`;
- `_locate_source` prefers the stable job-owned path and retains the old
  `UPLOAD_DIR` lookup only for existing jobs.

Command changes:

```python
cmd.extend([
    "-o", job_output_dir,
    "--project-dir", job_output_dir,
    "--job-id", job_id,
    "--managed-job",
    "--execution-generation", str(execution_generation),
])
```

Do not persist the raw command as the resumability source of truth. Reconstruct
it from `.job-config.json`.

Add two pure helpers near `child_env`:

```python
def build_job_command(job_config: dict) -> list[str]:
    ...


def build_job_child_env(job_config: dict, runtime_secrets: dict) -> dict[str, str]:
    ...
```

Both initial execution and recovery use these helpers. Unit tests compare their
outputs for a newly accepted job and the same job after restart.

Failure rule:

- if required job config, state, source ownership, resume manifest or encrypted
  required-secret persistence fails, remove the unstarted job directory,
  release any reservation and return an error;
- an accepted job never exists only in memory.

#### Resume manifest

Current anchors:

- `_RESUME_FILE`
- `_manifest_path`
- `_read_manifest`
- `_touch_manifest`
- `_write_resume_manifest`
- `_clear_resume_manifest`

Change:

- retain filenames and high-level responsibility;
- replace direct JSON writes with `job_state` functions;
- remove the raw `cmd` field;
- remove plaintext `webhook_secret`;
- persist only stable job identity, owner, reservation, priority, attempt
  count, heartbeat, instance, partial-processing information and references to
  encrypted secret records;
- make `_touch_manifest` an atomic read-modify-write guarded by the job lease
  or a dedicated manifest update lock;
- if a valid old manifest exists and a new write fails, preserve the old file.

#### Interrupted-job recovery

Current anchor: `_resume_interrupted_jobs`.

Change:

1. load `.resume.json`, `.job-config.json` and `.job-state.json`;
2. reject mismatched job IDs or configuration revisions;
3. observe existing heartbeat handover rules;
4. do not enqueue if an OS job lease is held;
5. reconstruct command and child environment from immutable job config;
6. recover required credentials from the encrypted record;
7. if credentials are unavailable, persist `waiting_for_credentials`;
8. if a private/signed source is unavailable, persist `waiting_for_source`;
9. enqueue eligible work without changing `execution_generation`;
10. never treat metadata-file existence alone as proof of completion.

Execution generation is allocated only after the eventual worker acquires the
lease.

#### Queue ownership

Current anchors:

- `process_queue`
- `run_job_wrapper`
- `_running_jobs`

Change:

- add one helper:

```python
def claim_job_execution(job_id: str, job: dict) -> tuple[JobLease, int]:
    """Acquire lease, increment generation atomically, persist processing."""
    ...
```

- call it from `run_job_wrapper` after semaphore admission but before
  `_running_jobs.add`, heartbeat, command construction or subprocess start;
- retain the lease in the in-memory job record through the subprocess,
  terminal-state persistence and terminal actions;
- if the lease is busy, release the semaphore, leave the manifest intact and
  let the owning instance proceed;
- release the lease in the outermost `finally`, after terminal state is
  durable;
- do not make a `--managed-job` subprocess acquire the same lease.

`JobLease.acquire()` receives project/job identity only. The helper reads the
current generation under the lock, increments it, writes `.job-state.json`,
updates the manifest heartbeat/owner and only then calls
`build_job_command()`.

#### Subprocess execution and markers

Current anchors:

- `run_job`
- `enqueue_output`

Change:

- reconstruct `cmd` and `env` from saved config at run time;
- recognize and persist a current-generation plan marker:

```text
CLIP_PLAN_READY <execution-generation> <metadata-filename> <plan-hash> <count>
```

- use the existing `CLIP_READY <index> <filename>` marker for both newly
  rendered and reused completed artifacts;
- use stable metadata filename instead of `glob("*_metadata.json")`;
- keep robust reads because metadata can appear while the child is active.

On `CLIP_PLAN_READY`, atomically record `live_plan` in `.job-state.json`. This
is a current-attempt handoff, not a reusable compute checkpoint.

After the child exits, drain and join the stdout reader before reading
`live_plan`, ready markers, metadata or choosing terminal status:

```python
await asyncio.to_thread(t_log.join, LOG_DRAIN_TIMEOUT_SECONDS)
if t_log.is_alive():
    raise RuntimeError("pipeline output reader did not drain")
```

Without this barrier, `run_job()` can race the final `CLIP_PLAN_READY` or
`CLIP_READY` lines still buffered in the reader thread.

#### Render-result validation and terminal state

Current anchors:

- `_clips_actually_rendered`
- `run_job`
- `run_job_wrapper`
- `_presented_status`
- `get_status`

Change:

- open the existing project with
  `CheckpointStore.open_existing_project(...)`;
- load the current routed clip plan using the shared
  `expected_clip_plan_key()` builder;
- call `valid_render_artifacts()` so app and `main.py` use identical source,
  plan, source-analysis and render-key validation;
- if checkpointing is disabled or `clip-plan.json` could not be persisted,
  validate the stable metadata file against the current execution generation,
  `live_plan.plan_hash`, count and `CLIP_PLAN_READY` handoff; this supplies the
  planned set for the current attempt only and is never reused to skip
  selection after restart;
- for checkpoint-aware jobs, trust only current-run ready markers, validated
  render checkpoints or the validated terminal artifact ledger;
- remove filename-existence fallback for checkpoint-aware jobs;
- after child exit, build planned, delivered and missing clip sets;
- set `completed` when all planned clips are valid, `partial` when at least one
  is valid after allowed retries, and `failed` when none is valid;
- include `missing_clip_indices` in partial results;
- call `commit_terminal_state()` once to atomically persist terminal status,
  internally allocated terminal revision, artifact ledger, missing indices and
  the full applicable terminal-action outbox derived by
  `required_terminal_actions()`;
- clear `.resume.json` only after that single atomic terminal commit succeeds.

There is no permitted intermediate state containing a terminal decision
without its outbox. A crash before the commit leaves a resumable compute job; a
crash after it leaves a terminal job whose pending actions are discoverable.

#### Retry cleanup

Current anchors:

- `_clean_for_retry`
- `_schedule_auto_retry`

Change:

- change the signature to accept the job ID/config context rather than only a
  directory:

```python
def _clean_for_retry(job_id: str, job: dict) -> None:
    ...
```

- open the checkpoint project, load the current plan and call
  `valid_render_artifacts()` before deleting anything;
- preserve files referenced by compatible valid render checkpoints;
- preserve stable source, job config, job state, resume state, checkpoints,
  metadata and terminal ledger;
- remove abandoned temp cuts and incomplete output only;
- reset in-memory `ready_files`, then allow `main.py` to re-emit markers for
  cached renders;
- do not delete all `*_metadata.json` files indiscriminately.

#### Terminal actions

Current anchors:

- `_settle_reservation`
- `_archive_managed_job`
- `_notify_job_webhook`
- `_autopilot_job_finished`
- `_notify_clips_ready`
- `_record_job_alert`
- `_track_proxy_usage`
- `_notify_clip_activity`

Change:

- treat `partial` as delivery of its validated subset;
- before the first side effect, require the outbox already written by
  `commit_terminal_state()`;
- replace the direct sequence in `run_job_wrapper` with:

```python
async def run_terminal_actions(job_id: str) -> None:
    ...


async def recover_pending_terminal_actions() -> None:
    ...
```

- `run_terminal_actions()` acquires the job lease when it is not already held,
  processes `reservation`, `archive`, `autopilot_finalize`,
  `autopilot_email`, `autopilot_publish`, `webhook`, `alert`, `proxy_usage`,
  `clips_ready_email`, and `clip_activity` in that order, and persists every
  transition;
- idempotent actions transition `pending|running -> running -> done` and may be
  retried after a crash;
- `_deliver_webhook` returns a delivery outcome instead of being detached with
  `asyncio.create_task`; the outbox marks `done` only after 2xx, includes a
  stable `event_id` in the body and sends `X-OpenShorts-Event-ID`;
- webhook semantics are at least once, so consumer deduplication by event ID is
  documented;
- email and Telegram actions mark `dispatching` atomically before calling
  providers that have no idempotency key. A recovered `dispatching` action
  becomes `manual_review` rather than sending a possible duplicate;
- bounded best-effort alert/analytics failure is persisted but does not change
  terminal media status;
- `alert`, `clips_ready_email`, and `clip_activity` use
  `dispatching -> done`; recovery of `dispatching` moves to `manual_review`
  rather than automatically duplicating a provider call;
- make proxy accounting idempotent before retrying it from the outbox: add a
  stable unique event ID to `cloud.models.ProxyUsage`, pass it through
  `cloud.proxy_ledger.record_download`, insert-once, and increment in-memory
  counters/send proxy alerts only when the row was newly created;
- settle billing according to the design policy and rely on the ledger's
  guarded state transition for idempotency;
- archive only validated delivered files using existing database upserts;
- remove the current self-host `upload_job_artifacts()` call from `run_job()`;
  self-host S3 backup is the `archive` outbox action with mode
  `self_host_s3`, while managed R2 uses mode `managed_r2`;
- emit a distinct partial webhook/event with missing clip indices;
- exclude checkpoint, job-state, resume and secret files;
- ensure retry counters stop after persisted terminal state;
- startup calls `recover_pending_terminal_actions()` after disk job recovery,
  without rerunning `main.py`.

After terminal state and outbox are durable, release the GPU/concurrency
semaphore and mark the compute queue item done before waiting on network
actions. Keep or reacquire only the per-job lease for outbox transitions. This
prevents a slow archive or webhook from occupying a video-processing slot.

`terminal_revision` increments exactly once when the terminal decision is first
persisted. Derive each stable event ID as the SHA-256 of canonical JSON
containing `openshorts-terminal-v1`, `job_id`, `terminal_revision`, and action
name. Allowed action states are:

```text
pending
running
dispatching
done
not_required
permanent_failure
manual_review
```

Recovery policy by action:

| Action | Recovery semantics |
|---|---|
| `reservation` | Retry `running`; ledger status guard makes commit/release idempotent. |
| `archive` | Retry `running`; job/clip upserts and deterministic object keys are idempotent. |
| `autopilot_finalize` | Retry `running`; DB transaction records outcome and a durable publish/email plan and returns the same plan on replay. |
| `autopilot_email` | Per-message `dispatching`; recovered ambiguity becomes `manual_review`. |
| `autopilot_publish` | Per-clip publish item uses `dispatching`; recovered ambiguity becomes `manual_review` to avoid duplicate social posts. |
| `webhook` | Retry `running`; at-least-once with stable event ID for receiver dedupe. |
| `proxy_usage` | Retry `running` only after unique event-ID persistence is implemented. |
| `alert` | `dispatching` after crash becomes `manual_review`; no blind duplicate. |
| `clips_ready_email` | `dispatching` after crash becomes `manual_review`; no blind duplicate. |
| `clip_activity` | `dispatching` after crash becomes `manual_review`; no blind duplicate. |

Autopilot publish item schema:

```json
{
  "item_id": "<64-char-sha256>",
  "event_id": "<64-char-sha256>",
  "clip_key": "<render/plan clip identity>",
  "clip_index": 0,
  "profile_id": "<non-secret connected-profile reference>",
  "platforms": ["youtube"],
  "scheduled_at": "<UTC ISO-8601>",
  "state": "pending",
  "attempts": 0,
  "last_error_code": null
}
```

One item represents the existing single `_upload_clip` request for one clip and
its validated ordered platform list. `item_id` hashes terminal event ID, clip
key, profile reference, platforms, and scheduled time. It contains no access
token. `transition_terminal_action_item()` performs an atomic expected-state
compare-and-set and recomputes the parent action:

- `done` when every item is `done` or `not_required`;
- `manual_review` when any item is `manual_review`;
- `permanent_failure` when no item is ambiguous and at least one exhausted
  item is `permanent_failure`;
- otherwise `pending` or `running`.

#### Disk recovery

Current anchors:

- `_recover_jobs_from_disk`
- `_job_view_from_disk`

Change:

- prefer `.job-state.json`, including its terminal artifact ledger;
- revalidate every terminal artifact hash and media probe before serving;
- recover `completed`, `partial`, `waiting_for_credentials`,
  `waiting_for_source`, `queued` and `processing`;
- use legacy metadata discovery only for pre-checkpoint jobs;
- use `<job_id>_metadata.json` for checkpoint-aware jobs.
- treat `.job-state.json` as owner authority; when a legacy `.owner` projection
  disagrees, reject access/recovery and log an integrity error rather than
  choosing one silently;
- enqueue unfinished terminal outbox actions separately from compute resume.

#### Cleanup

Current anchors:

- `_job_is_active`
- `_sweep_expired_job_dirs`
- `_sweep_expired_uploads`
- `_enforce_uploads_size_cap`
- `_enforce_output_size_cap`
- `_sweep_retained_sources`
- `cleanup_jobs`
- `_purge_local_jobs_for_user`

Change:

- deleting a job directory removes checkpoints, state, encrypted resume secret
  and retained source together;
- do not delete a project while its lease is held;
- stop treating new checkpoint-aware sources as independent `UPLOAD_DIR`
  objects;
- retain old upload cleanup for legacy jobs and pending upload slots;
- account for nested checkpoint bytes in size-cap decisions;
- forget in-process lease records after deletion.

### 2. `main.py`

#### Import order

Current anchor: top-level imports and `load_dotenv()`.

Change:

- call `load_dotenv()` before importing `layout_picker`, `llm_backend`,
  `vision_backend` or any module that reads environment values;
- import `checkpoint_store` and `job_state` after dotenv;
- removing provider import-time environment reads remains desirable.

#### CLI and project initialization

Current anchor: `if __name__ == "__main__"` and argument construction.

Change:

1. add the four internal/project arguments defined above;
2. resolve output and persistent project directory before source ingest;
3. determine whether the run is app-managed or standalone;
4. for a checkpoint-enabled standalone run, create the project directory,
   choose/load its job ID and acquire `JobLease` before creating or replacing
   any project state;
5. load or create job config under that lease;
6. validate job ID, configuration revision and execution generation;
7. create the null or enabled unbound `CheckpointStore`;
8. resolve or ingest and verify the stable source;
9. call `store.bind_source(source_identity)`;
10. release a standalone lease through `finally`/`atexit`.

For `--managed-job`, `app.py` already holds the lease before starting the
process; `main.py` validates the saved generation but does not reacquire the
same lock.

Do not create checkpoints in the current working directory.

#### Source ingest

Current anchors:

- `download_youtube_video`
- `cap_source_duration`
- the URL/local-input branch after argument parsing

Change:

- extend `download_youtube_video` with an optional stable destination:

```python
def download_youtube_video(
    url,
    output_dir=".",
    on_audio=None,
    destination_path=None,
):
    ...
```

- reuse a validated retained source before calling yt-dlp;
- do not delete a validated retained destination merely because it exists;
- for a new URL source, download to a temporary path, apply cap/range,
  media-probe and hash final bytes, atomically place the stable source, then
  save `source.json`;
- for a local upload, verify the job-owned source and save the same checkpoint
  shape;
- checkpoint only after final post-cap source identity is known;
- do not publish early-ASR checkpoint output until source identity is final.

The source checkpoint contains source hash, byte size, final duration,
stream/probe summary, retained relative path and cap/range identity.

#### Legacy transcript checkpoint removal

Current anchors:

- `TRANSCRIPT_CHECKPOINT`
- `_checkpoint_source_key`
- `save_transcript_checkpoint`
- `load_transcript_checkpoint`
- `clear_transcript_checkpoint`

Change:

- delete these after ASR checkpointing is connected;
- ignore and eventually clean old `.transcript_checkpoint.json`;
- do not migrate it because filename plus duration is insufficient identity.

#### ASR

Current anchors:

- `transcribe_video`
- the `args.transcript`, early transcript and `NoAudioError` block
- `speech_is_sparse`
- `transcribe_backends.transcribe_media`

Change:

1. resolve the requested ASR configuration and allowed fallback routes;
2. load `asr.json` through `load_routed_fixed()` so its recorded effective
   backend route is validated without rerunning transcription;
3. represent three completed outcomes: `speech`, `no_audio` and
   `sparse_speech`;
4. call `transcribe_video` only on a miss;
5. validate complete transcript shape before saving;
6. save exactly one complete ASR outcome;
7. route cached no/sparse speech to visual selection without rerunning ASR.

For `--transcript`, seal the retained transcript SHA-256 in the source
configuration, validate the complete transcript against the final source
duration, and use the routed `precomputed-transcript` ASR checkpoint. A changed
or missing transcript is not reusable.

#### Source layout

Current anchors:

- `layout_picker.ENABLED`
- `layout_picker.pick_and_apply`
- the layout block before subtitle detection

Change:

- resolve layout config at job start;
- checkpoint both selected layout and deliberate no-layout;
- separate decision from mutation:

```python
decision = layout_picker.pick(...)
layout_picker.apply(decision)
```

- on reuse, apply the cached decision as a fresh call would;
- include source dimensions, sampled-frame hashes, provider/model identity,
  prompt/schema versions and available layout features in compatibility;
- embed the exact completed or best-effort fallback decision in the final clip
  plan. On plan reuse, restore that embedded decision before consulting the
  standalone layout checkpoint or calling the provider.

#### Burned-subtitle detection

Current anchors:

- lazy `import burned_subtitles`
- `burned_subtitles.detect`
- `source_subtitle_band`

Change:

- checkpoint both detected band and valid `none`;
- include sample hashes, detector identity and threshold configuration;
- cached `none` must not be confused with a cache miss;
- embed the exact band/none/failure fallback used by rendering in the final
  clip plan and restore it directly when that plan is reused.

#### Source scene detection

Current anchors:

- `vision_backend._scene_boundaries`
- `vision_backend.select_clips`

Change:

- compute or load one source-scene result before local visual windows;
- pass it into local visual selection instead of letting
  `vision_backend._scene_boundaries` recompute it;
- save boundaries, fps, effective detector and fallback provenance;
- fingerprint the effective engine if TransNetV2 falls back to PySceneDetect.

This source-level result is not the same as the per-cut scene detection used by
`render_clip`, `reframe_v2` or the clip editor. Those remain render work and
enter render compatibility separately.

#### Transcript score windows

Current anchors:

- `get_viral_clips`
- `_run_stage_split`
- `build_transcript_windows`
- `score_batches`
- `reconcile_scored_windows`

Change:

- construct deterministic windows before model calls;
- derive a unit key from canonical timestamped words;
- load score units before call batches;
- call the provider only for misses;
- after a complete response, save each independently valid score, including a
  deliberate valid empty/low-score result;
- do not save a batch placeholder implying all windows completed;
- reconcile cached and fresh scores through existing logic;
- count cost only for fresh calls.

#### Transcript detail windows

Current anchors:

- `_run_local_detail_window`
- `_run_local_detail_windows`
- local-provider branch inside `get_viral_clips`

Change:

- load one detail checkpoint before each local one-window call;
- validate, normalize and save each completed window independently;
- distinguish reusable explicit empty from non-reusable all-invalid or failed
  calls;
- preserve candidate-floor recovery using cached and fresh windows;
- leave recursive/batched Gemini detail outside version 1 fine-grained detail
  checkpointing.

#### Local global ranking

Current anchors:

- `_rank_local_candidates`
- `apply_global_candidate_ranking`

Change:

- load/save one `text-rank.json`;
- fingerprint the ordered normalized candidate set and ranking config;
- save only a successfully validated model ranking;
- when ranking fails and the pipeline deliberately keeps deterministic input
  order, do not write `text-rank.json`; record `ranking_fallback` only in the
  final clip-plan provenance.

#### Visual selection

Current anchor: `get_visual_clips`.

Change:

- pass checkpoint store, source scenes and resolved config to
  `vision_backend.select_clips`;
- preserve Gemini fallback;
- implement local fine-grained visual windows inside `vision_backend`;
- save a whole-video Gemini result only when immutable model revision can be
  recorded; otherwise disable cross-process model-output reuse.

#### Final clip plan

Current anchor: after `get_viral_clips`/`get_visual_clips`, before metadata.

Change:

- immediately after source binding, read `clip-plan.json` through
  `load_routed_fixed()` using a route-context validator for selection route,
  ASR effective route, and ASR outcome;
- calculate the current expected transitive key for that context from source
  identity and immutable job configuration; do not require the ASR payload or
  rerun ASR merely to rediscover those routes;
- restore the plan's embedded source-layout and burned-subtitle decisions
  before any optional provider call, so resumed renders use the same source
  analysis as the completed plan;
- use a transitive compatibility key;
- validate boundaries, durations, required metadata and deduplication;
- save only final normalized ordered plan;
- write `<job_id>_metadata.json` atomically with `job_id`,
  `execution_generation`, plan hash, planned clip count, and clip-plan
  compatibility key when one was persisted;
- immediately print `CLIP_PLAN_READY` after the metadata rename succeeds;
- do not use `video_title` for metadata filename.

#### Per-clip render

Current anchors:

- nested `_process_one_clip`
- `clip_render_order`
- `ThreadPoolExecutor` submission
- failed-clip retry loop

Change:

1. move `_process_one_clip` to module scope or pass an immutable render
   context containing the checkpoint store;
2. derive stable `clip_key` from clip plan and render compatibility;
3. load and validate render checkpoints before submitting work;
4. emit `CLIP_READY` for reusable renders;
5. submit only misses;
6. render through cut, reframe, hook, captions, watermark and final probe;
7. hash/probe the served artifact;
8. save render checkpoint before `CLIP_READY`;
9. if save fails but current artifact is valid, emit ready and mark it
   non-reusable;
10. retry only missing failures.

Stable names should use the clip key, for example:

```text
clip-<first-16-hex-of-clip-key>.mp4
```

Do not infer checkpoint validity from filename existence.

#### Metadata and final cleanup

Current anchors:

- direct metadata `json.dump` calls;
- `clear_transcript_checkpoint`;
- URL-source deletion at the end of `main.py`.

Change:

- write metadata atomically;
- persist render-added fields atomically after workers;
- remove legacy transcript clear;
- retain job-owned source for checkpoint-enabled projects until project
  cleanup;
- preserve current `--keep-original` semantics for checkpoint-disabled CLI.

### 3. `transcribe_backends.py`

Current anchors:

- `transcribe_media`
- `run_whisper_transcription`
- `_transcribe_with_whisper`
- `_transcribe_with_parakeet`
- `_parakeet_fallback_reason`

Add a provenance-returning entry point while preserving existing callers:

```python
def transcribe_media_with_provenance(media_path):
    """Return {outcome: speech|no_audio, transcript, provenance}."""
    ...


def transcribe_media(media_path):
    result = transcribe_media_with_provenance(media_path)
    if result["outcome"] == "no_audio":
        raise NoAudioError(result["provenance"]["fallback_reason_detail"])
    return result["transcript"]
```

Where needed, add internal provenance-returning Whisper helpers rather than
changing the public `run_whisper_transcription()` tuple expected by existing
tests/callers.

The provenance object contains:

```json
{
  "requested_backend": "parakeet",
  "effective_route": "parakeet-to-whisper-cpu",
  "effective_backend": "whisper",
  "effective_model": "small",
  "immutable_model_revision": null,
  "device": "cpu",
  "compute_type": "int8",
  "fallback_reason_code": "parakeet_unsupported_language",
  "fallback_reason_detail": "<bounded diagnostic>",
  "attempts": [
    {"backend": "parakeet", "outcome": "rejected"},
    {"backend": "whisper", "device": "gpu", "outcome": "cuda_failed"},
    {"backend": "whisper", "device": "cpu", "outcome": "success"}
  ],
  "parameters_hash": "<canonical hash>",
  "implementation_version": "asr-backend-v1"
}
```

Use bounded route/reason enums; human-readable exception text is diagnostic
provenance and never a compatibility-key input. At minimum support:

```text
parakeet
parakeet-to-whisper-gpu
parakeet-to-whisper-cpu
whisper-gpu
whisper-cpu
no-audio
```

`main.transcribe_video()` consumes the new entry point directly and
returns/passes outcome, transcript and provenance. Sparse-speech classification
still occurs after a completed transcript in `main.py`, which changes the ASR
checkpoint outcome to `sparse_speech` while retaining backend provenance. This
is the only reliable source for effective fallback identity.

### 4. `scene_detection.py`

Current anchor: `detect_scenes`.

Add without breaking existing callers:

```python
def detect_scenes_with_provenance(video_path, resolved_config=None):
    """Return (scene_list, fps, provenance)."""
    ...


def detect_scenes(video_path):
    scene_list, fps, _ = detect_scenes_with_provenance(video_path)
    return scene_list, fps
```

Provenance includes requested/effective engine, model version, thresholds,
fallback reason and implementation version. Accept resolved config rather than
reading changing environment values midway through a job.

### 5. `layout_picker.py`

Current anchors:

- `_MODE`
- `SHADOW`
- `ENABLED`
- `SAMPLE_FRAMES`
- `SAMPLE_WIDTH`
- `pick`
- `pick_and_apply`

Change:

- replace import-time settings with `resolve_config(env=None)`;
- make `pick(video_path, video_duration, config)` return a serializable
  decision/provenance without mutating feature flags;
- add `apply(decision)` for mutations;
- retain `pick_and_apply` as a compatibility wrapper;
- expose deterministic sample descriptors/hashes for compatibility.

### 6. `burned_subtitles.py`

Current anchors:

- `ENABLED`
- `detect`

Change:

- replace import-time enablement with `resolve_config(env=None)`;
- allow `detect(video_path, frames=None, config=None)` to use immutable config;
- return result plus provenance, or add `detect_with_provenance` while keeping
  `detect` compatible;
- expose sample hashes and distinguish valid `none` from detector failure.

### 7. `vision_backend.py`

Current anchors:

- `select_clips`
- `_scene_boundaries`
- `_decode_samples`
- `_window_response`
- the per-window loop inside `select_clips`

Change the entry point to:

```python
def select_clips(
    video_path: str,
    video_duration: float,
    language: str,
    *,
    checkpoint_store=None,
    source_scenes=None,
    resolved_config=None,
):
    ...
```

Per window:

1. derive deterministic boundaries;
2. decode deterministic frame timestamps;
3. hash actual encoded JPEG request bytes;
4. build semantic input hash from timestamps, JPEG hashes, prompt/schema,
   source scenes and model identity;
5. load keyed visual checkpoint;
6. call Ollama only on a miss;
7. validate/normalize candidates;
8. save valid candidates or deliberate valid empty;
9. do not save all-invalid, failed or interrupted calls;
10. continue other windows when one yields no candidates.

Final global dedupe/rank remains deterministic and enters clip-plan
compatibility.

### 8. `llm_backend.py`

Current anchors:

- provider/base URL/model helpers;
- `describe`
- `generate_json`

Change:

- add `resolved_descriptor()` returning provider, normalized endpoint class,
  requested model and immutable model digest/revision;
- query Ollama model metadata once per job and record its digest;
- if immutable identity is unavailable, mark fine-grained/intermediate
  model-call checkpoints non-reusable across process restart; completed final
  clip plans remain reusable inside the same sealed job/source;
- keep checkpoint storage out of this module.

### 9. `s3_uploader.py`

Current anchor: `upload_job_artifacts`.

Change:

- replace broad JSON discovery with an explicit artifact allowlist;
- change the function contract to:

```python
def upload_job_artifacts(
    directory: str,
    job_id: str,
    terminal_artifacts: list[dict],
    metadata_filename: str,
) -> None:
    ...
```

- call it only from the terminal `archive` outbox action after
  `commit_terminal_state()`, never directly from `run_job()`;
- allow stable metadata and validated delivered media only;
- exclude dotfiles, `.checkpoints/`, temps, locks, resume state, encrypted
  secrets, job config/state, terminal ledger and legacy transcript checkpoint;
- test that nested sensitive files cannot be uploaded.

### 10. `cloud/videos.py`

Current anchor: `archive_job`.

Change:

- consume validated terminal artifacts supplied by `app.py`;
- archive only stable metadata and delivered media;
- never scan/upload checkpoint or state directories;
- support `partial` with missing-clip list;
- do not restore transient checkpoints from archive.

### 11. `cloud/autopilot.py`

Current anchors:

- `on_job_finished`
- `_publish`
- `_update_run`

Change:

- split `on_job_finished` into three resumable operations:

```python
async def finalize_job(job_id: str, job: dict, failure_reason: str | None) -> dict:
    """Idempotently persist outcome and return the same durable action plan."""
    ...


async def send_completion_email(plan: dict) -> None:
    ...


async def publish_one(plan_item: dict) -> bool:
    ...
```

- `finalize_job` performs the guarded outcome update and persists or reloads a
  deterministic email/publish plan in the same database transaction;
- replay after the run already left `PROCESSING` returns the stored plan rather
  than no-oping;
- remove detached `asyncio.create_task(_publish(...))` from the terminal path;
- after `autopilot_finalize`, atomically fill the `autopilot_email` and
  per-clip `autopilot_publish.items[]` outbox details, marking them
  `not_required` where appropriate;
- construct every publish item with the exact nested schema defined in the
  terminal-action section and derive item/event IDs deterministically;
- each publish item transitions to `dispatching` before `_upload_clip`;
- a successful item becomes `done`; a definite pre-request failure may retry;
  a recovered `dispatching` item becomes `manual_review` so a social post is
  not duplicated;
- `_update_run` derives `posted_count` from durable completed publish items.

If adding the durable action plan requires new columns/table, include that
schema update in the same commit and keep the job-state outbox identifiers
linked to the database plan.

### 12. `cloud/models.py`

Current anchor: `ProxyUsage`.

Change:

- add a nullable unique `event_id` column for backward compatibility with old
  rows;
- add the repository's normal database migration/schema-update path for that
  column and unique index;
- terminal outbox proxy accounting always supplies its stable event ID.

### 13. `cloud/proxy_ledger.py`

Current anchors:

- `_persist`
- `record_download`

Change:

- accept `event_id`;
- insert with duplicate-event protection and return whether a row was newly
  created;
- only a new row updates `_proxy_month` or sends a paid-proxy alert;
- a duplicate event returns success to the outbox without double counting.

### 14. `.env.example`

Add:

```dotenv
# Reuse fully completed pipeline stages after an interrupted/retried job.
# Disabled by default until explicitly enabled.
PIPELINE_CHECKPOINTS_ENABLED=false
```

Also document the optional resume-secret keyring variables from the encrypted
secret section. They are needed only when request-scoped secrets must survive a
restart. Schema and implementation versions remain code constants.

### 15. `requirements.txt`

Add and pin the `cryptography` package and use
`cryptography.hazmat.primitives.ciphers.aead.AESGCM` for AES-256-GCM in
`.resume-secrets.enc`. Add a round-trip, wrong-key, wrong-AAD and
tampered-ciphertext test before the dependency is used by `app.py`.

### 16. Documentation

Update:

- `docs/runbooks/local-ollama-windows-setup.md`
- the repository `README.md` checkpoint/restart section

Document enable/disable values, default-disabled behaviour, restart commands,
persistent output volume, disk usage, safe project deletion, invalidation
behaviour and Windows Docker verification commands.

## Test Map

### New test files

Create:

```text
tests/test_job_state.py
tests/test_checkpoint_store.py
tests/test_source_checkpoint.py
tests/test_asr_checkpoint.py
tests/test_source_analysis_checkpoints.py
tests/test_selection_checkpoints.py
tests/test_render_checkpoints.py
tests/test_checkpoint_recovery.py
tests/test_terminal_action_outbox.py
scripts/test_checkpoint_resume.ps1
```

### Existing tests to extend

| Existing test | Required coverage |
|---|---|
| `tests/test_transcript_checkpoint.py` | Replace filename/duration checks with source-byte-bound ASR outcomes. |
| `tests/test_auto_retry.py` | Preserve compatible completed renders; remove incomplete files only. |
| `tests/test_deploy_drain.py` | Lease ownership, stale takeover and atomic heartbeat. |
| `tests/test_process_handover.py` | Immutable config reconstruction and execution generation. |
| `tests/test_cleanup_live_jobs.py` | Do not delete leased projects; delete nested state after release. |
| `tests/test_account_local_purge.py` | Remove source, checkpoints and encrypted resume secrets. |
| `tests/test_media_path_guard.py` | Deny dotfiles and `.checkpoints` through media routes. |
| `tests/test_clip_ready_marker.py` | Cached and fresh renders emit identical markers. |
| `tests/test_phantom_clips.py` | No filename-existence fallback for checkpoint jobs. |
| `tests/test_partial_processing.py` | Partial terminal result and missing clip list. |
| `tests/test_source_access.py` | Stable job-owned source plus legacy fallback. |
| `tests/test_layout_picker.py` | Resolved config, cached decision and cached no-layout. |
| `tests/test_burned_subtitles.py` | Cached band, cached none and failure distinction. |
| `tests/test_vision_backend.py` | Per-window hit/miss/empty/all-invalid and continue-after-empty. |
| `tests/test_local_llm_one_window.py` | One detail checkpoint per canonical transcript window. |
| `tests/test_local_llm_batch_recovery.py` | Mixed cached/fresh windows and floor recovery. |

Additional required assertions:

- direct routed clip-plan load works when `asr.json` is absent;
- unknown or malformed route context is rejected before expected-key
  calculation;
- ASR provenance records Parakeet rejection, Whisper GPU failure and effective
  CPU success through the actual fallback path;
- two recovery scanners may enqueue the same stale manifest, but only one
  lease owner allocates a generation and starts `main.py`;
- `app.py` and `main.py` return the same valid render set from identical state;
- checkpoint-disabled and checkpoint-save-failed runs validate their current
  planned set through generation-bound metadata plus `CLIP_PLAN_READY`;
- a crash before `commit_terminal_state()` leaves compute resumable, while a
  crash immediately after it recovers every pending action from the atomic
  outbox;
- a crash between every terminal action and its `done` write follows the
  documented idempotent, at-least-once or manual-review policy;
- replaying the same proxy action event ID creates one database row, one byte
  count and at most one alert;
- terminal validation cannot begin until the subprocess stdout reader has
  drained its final plan/clip markers;
- terminal commit rejects an execution-generation mismatch, allocates the
  terminal revision itself, and cannot omit a required action;
- Autopilot finalize replay returns the same durable plan, and process death
  during each email/publish item produces `manual_review` rather than a silent
  loss or duplicate automatic post;
- Autopilot item compare-and-set rejects stale state, produces stable IDs, and
  recomputes parent completion exactly as documented;
- self-host S3 backup starts only after atomic terminal commit and receives
  only the validated metadata/artifact allowlist.

### Interruption matrix

Terminate after each completed boundary, restart the same project, and assert:

| Boundary | Expected reuse |
|---|---|
| Retained source committed | No re-upload, re-download or recapping. |
| ASR completed | No ASR model call. |
| Layout completed | Cached decision applied without provider call. |
| Burned-subtitle check completed | No detector call. |
| Source scenes completed | No scene detector call. |
| Some text score windows completed | Call missing score windows only. |
| Some local detail windows completed | Call missing detail windows only. |
| Local rank completed | Reuse ranking. |
| Some local visual windows completed | Call missing visual windows only. |
| Final clip plan completed | No upstream selection calls. |
| Some renders completed | Restore ready markers; render missing clips only. |
| All renders valid before terminal actions | Recover media once and resume pending terminal actions without rerunning compute. |

Also run a full disabled pipeline and assert that no `.checkpoints/` directory
or compute checkpoint is written.

## Ordered Commit Plan

Each item is a separate reviewable commit.

1. **Durable job state and ownership** — add `job_state.py`; modify `app.py`
   and `requirements.txt`; add job-state, encryption, and handover tests.
2. **Checkpoint store foundation** — add `checkpoint_store.py`; modify
   `.env.example` and import order; add store tests.
3. **Stable retained source** — modify `app.py` and `main.py`; add source,
   cleanup and access tests.
4. **ASR outcomes and provenance** — modify `main.py` and
   `transcribe_backends.py`; remove legacy transcript helpers; replace
   transcript checkpoint tests.
5. **Source analysis** — modify `main.py`, `scene_detection.py`,
   `layout_picker.py`, `burned_subtitles.py`; add source-analysis tests.
6. **Local transcript selection** — modify `main.py`, `llm_backend.py`; add
   score/detail/rank checkpoint tests.
7. **Visual selection** — modify `main.py`, `vision_backend.py`; extend vision
   tests.
8. **Final clip plan and metadata** — modify `main.py`, `app.py`; extend
   metadata/recovery tests.
9. **Render checkpoints** — modify `main.py`, `app.py`; add render, marker,
   retry and phantom-clip tests.
10. **Terminal recovery and action outbox** — modify `app.py`,
    `cloud/autopilot.py`, `cloud/models.py`, and `cloud/proxy_ledger.py`; add
    schema updates, terminal artifact, Autopilot sub-action, idempotent
    proxy-accounting, and outbox crash-recovery tests.
11. **Retention and runbook** — modify `app.py`, `s3_uploader.py`, and
    `cloud/videos.py`; add cleanup/archive/Windows tests and documentation.

Do not start provider or render checkpointing before commits 1–3 establish
identity, source reuse and exclusive ownership.

## Implementation Readiness Checklist

- [x] Feature flag and default are defined.
- [x] Completed versus partial work is defined per stage.
- [x] Compatibility and transitive invalidation are defined.
- [x] Crash-safe storage and single-writer ownership are defined.
- [x] Request-specific configuration survives restart.
- [x] Secrets are excluded from plaintext checkpoint state.
- [x] Source, ASR, analysis, selection, plan and render hooks map to current
  functions.
- [x] Retry, terminal, archive and cleanup changes are mapped.
- [x] New modules and responsibilities are defined.
- [x] Existing and new tests are mapped.
- [x] Work is divided into dependency-ordered commits.

The mapped implementation is complete in the working branch. It still requires
the documented Windows Docker kill/restart matrix before release. Any code
change that conflicts with the architecture design must update and review the
design first rather than silently diverging from it.
