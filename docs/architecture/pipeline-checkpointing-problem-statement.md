# OpenShorts Pipeline Checkpointing — Problem Statement

## Background

OpenShorts processes a source video through several compute-intensive branches:

```mermaid
flowchart TD
    A[Source ingest and validation] --> B[ASR stage: Parakeet or Whisper]
    A --> C[Source analysis needed by layout or silent-video selection]
    B --> D{Meaningful speech?}
    D -->|Yes| E[Transcript windows]
    E --> F[Gemini or local Qwen selection]
    D -->|No| G[Gemini whole-video vision or local Qwen-VL windows]
    C --> G
    F --> H[Validated structured clip plan]
    G --> H
    H --> I[Per-clip cut, reframe, tracking, hooks and subtitles]
    I --> J[Rendered MP4 clips]
```

Source downloading, ASR, source/scene analysis, LLM inference, and video
rendering can each consume substantial time, CPU, GPU, network quota, and
memory. A long video may take many minutes to progress through these branches.

## Problem

If processing is interrupted after a stage has completed—for example by stopping Docker, restarting the laptop, rebuilding a container, an application error, or a later pipeline stage failing—OpenShorts may repeat work that was already completed successfully.

Examples include:

- ASR finishes transcribing a 30-minute video, but clip selection later fails.
- Several Qwen analysis requests complete, but a later request or validation step fails.
- Some clips finish rendering, but another clip causes the job to fail.
- A silent-video analysis fails after the application has already confirmed that the video contains no meaningful speech.
- A URL source finishes downloading, but a later stage fails and a restart downloads it again.

Repeating these completed operations wastes processing time and compute. It can also produce different LLM results when a model is called again, because model responses are not guaranteed to be identical between runs.

## Required Behaviour

OpenShorts should preserve a checkpoint only after a pipeline stage has fully completed and its output has been validated.

An incomplete stage must never be treated as completed:

```text
Stage starts
    ↓
Stage is interrupted or returns invalid output
    ↓
Do not create a checkpoint
```

```text
Stage starts
    ↓
Stage completes successfully
    ↓
Validate the complete output
    ↓
Save the completed output as a checkpoint
```

For ASR, the completed timestamped transcript is the checkpoint. “ASR
completed” and “timestamped transcript created” are not two independent
checkpoints; they form one successful stage boundary.

A valid completed ASR outcome may be:

- A timestamped transcript containing meaningful speech.
- A completed result confirming that the video has no audio.
- A completed result confirming that the audio contains too little meaningful speech and the video must use visual analysis.

If ASR is interrupted before reaching one of these completed outcomes, no ASR
checkpoint should be recorded.

The same rule applies to independently resumable units. For example, one fully
validated Qwen window or one fully rendered clip may be checkpointed even while
other windows or clips are incomplete. A truncated/interrupted model transport
or partially rendered MP4 may not be checkpointed. A complete response that
contains several independently validated work units may checkpoint only those
validated units.

## Expected Resume Behaviour

For checkpointing, “the same project” means the same durable job/project ID and
source identity. An automatic retry or restart creates a new execution
generation under the same immutable configuration revision. A user-requested
configuration change creates a new configuration revision and invalidates the
affected stages.

When that project resumes, OpenShorts should:

1. Check whether a valid completed output exists for the next pipeline stage.
2. Confirm that the checkpoint belongs to the same source video and compatible processing configuration.
3. Reuse the checkpoint when it is valid.
4. Restart the stage normally when the checkpoint is missing, incomplete, corrupt, or incompatible.
5. Continue through the dependency graph using every compatible completed unit,
   rather than assuming the pipeline is one linear sequence.

For example:

```text
ASR completed and transcript checkpoint exists
    ↓
Docker or OpenShorts restarts
    ↓
Transcript checkpoint is validated
    ↓
ASR is skipped
    ↓
Processing resumes with clip analysis
```

## Desired Result

Checkpointing should prevent completed work from being unnecessarily repeated while ensuring that partial, corrupt, stale, or incompatible results are never reused.

The intended outcome is:

- Less repeated CPU and GPU work.
- Faster recovery after interruption or downstream failure.
- No reuse of partial model output.
- Consistent clip-selection results within a resumed project.
- Safe continuation from the latest fully completed pipeline stage.

## Scope of This Document

This document records only the checkpointing problem and required behaviour. It does not prescribe the storage format, implementation architecture, configuration variables, cleanup policy, or code changes.

The companion implementation design is
[Pipeline Checkpointing Design](pipeline-checkpointing-design.md), with exact
code insertion points in the
[Pipeline Checkpointing Implementation Map](pipeline-checkpointing-implementation-map.md)
and the end-to-end
[Actual Code Flow](pipeline-checkpointing-code-flow.md).
