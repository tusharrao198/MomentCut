import argparse
import json
import mimetypes
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from clip_selection import (clip_count_targets, clip_duration_bounds,
                            lookup_model_prices)

load_dotenv()


# --- Structured output schemas (passed as response_schema so the API
# --- guarantees the format instead of us repairing free-form JSON). ---

class ScoredWindowModel(BaseModel):
    id: str
    start: float
    end: float
    score: int
    reason: str


class ScoreResponse(BaseModel):
    windows: List[ScoredWindowModel]


class ModelResponseError(ValueError):
    """The model answered, but its response was not usable structured output."""


class ThumbnailBrief(BaseModel):
    """Optional per-clip creative facts usable by thumbnail generation."""
    subject: str = ""
    moment: str = ""
    emotion: str = ""
    visual_style: str = ""
    composition: str = ""
    thumbnail_text: str = ""
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class DetailClipModel(BaseModel):
    start: float
    end: float
    source_window_id: str
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str
    # One sentence on what makes THIS moment a clip, shown under the score in
    # the dashboard. Defaulted so a small local model that skips it does not
    # fail schema validation and lose the clip.
    why: str = ""
    youtube_description_intro: str = Field(
        default="",
        description=(
            "A factual, clip-specific YouTube description of 2-4 complete "
            "sentences and roughly 45-90 words, without hashtags or invented facts."
        ),
    )
    youtube_tags: list[str] = Field(default_factory=list)
    youtube_hashtags: list[str] = Field(default_factory=list)
    thumbnail_brief: ThumbnailBrief = Field(default_factory=ThumbnailBrief)


class DetailResponse(BaseModel):
    shorts: List[DetailClipModel]


class LocalWindowDetailClipModel(DetailClipModel):
    """One local transcript-window result; ownership is set by the server."""
    source_window_id: str = ""
    predicted_score: int = 0
    video_description_for_tiktok: str = ""
    video_description_for_instagram: str = ""
    video_title_for_youtube_short: str = ""
    viral_hook_text: str = ""


class LocalWindowDetailResponse(BaseModel):
    """A local window can yield one grounded clip or no clip."""
    clip: Optional[LocalWindowDetailClipModel] = None


class LocalRankedCandidateModel(BaseModel):
    candidate_id: str
    # Reconciliation clamps this to 0..100. Keeping the transport schema
    # permissive prevents one small-model arithmetic mistake from discarding
    # an otherwise complete global ranking.
    score: int
    reason: str = ""


class LocalClipRankResponse(BaseModel):
    candidates: List[LocalRankedCandidateModel]


# Visual (no-transcript) clip selection: Gemini watches a silent video and
# picks moments from the imagery. Same output shape as DetailClipModel minus
# the transcript-only source_window_id.
class VisualClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str
    why: str = ""
    youtube_description_intro: str = Field(
        default="",
        description=(
            "A factual, clip-specific YouTube description of 2-4 complete "
            "sentences and roughly 45-90 words, without hashtags or invented facts."
        ),
    )
    youtube_tags: list[str] = Field(default_factory=list)
    youtube_hashtags: list[str] = Field(default_factory=list)
    thumbnail_brief: ThumbnailBrief = Field(default_factory=ThumbnailBrief)


class VisualResponse(BaseModel):
    shorts: List[VisualClipModel]


VISUAL_PROMPT_TEMPLATE = """
You are a senior short-form video editor. This video has NO speech/audio — judge
it purely by what you SEE. Watch the whole thing and pick the {min_clips}–{max_clips} MOST engaging
visual moments for TikTok / Reels / Shorts (action, reveals, transformations,
striking or funny shots, satisfying payoffs, dramatic movement).

TIME CONTRACT — STRICT:
- Timestamps in ABSOLUTE SECONDS from the start (usable with ffmpeg -ss/-to).
- Only numbers with up to 3 decimals (e.g. 0, 12.5, 47.250).
- 0 <= start < end <= {video_duration}.
- Each clip {min_secs:g} to {max_secs:g} seconds long. If the whole video is
  shorter than {min_secs:g}s, return one clip spanning the full video.
- Cut on visual scene changes, never mid-motion.

For each clip write catchy copy in {language} (a scroll-stopping hook, a TikTok
and an Instagram description, and a YouTube title ≤100 chars). Also write a
truthful, clip-specific `youtube_description_intro` of 2-4 complete sentences
and roughly 45-90 words. Describe what happens, the key action or payoff, and
enough context for a cold viewer to understand the clip. Do not include
hashtags, links, calls to action, or facts that are not visible in the video.
Only name a game, mode, map, person, hardware, resolution, frame rate, or
recording setting when the video itself clearly establishes it. Order clips
best to worst by how likely they are to stop a viewer scrolling.
"""


# Grounded rewrite of hook + title for a clip whose meaning lives on screen
# (SCREENCAST / WIDE / INSET stretches): the detail pass never saw a frame,
# so its hook summarises the topic instead of naming what is being shown.
class GroundedHook(BaseModel):
    on_screen: str
    viral_hook_text: str
    video_title_for_youtube_short: str


GROUNDED_HOOK_PROMPT = """
These frames come from ONE short clip (the whole clip, in order) and the
transcript below is exactly what is said during it. Most of this clip's
meaning is on the screen, not in the face.

1. `on_screen`: one line naming what is shown — the app, window, product,
   document, code, chart or on-screen text — as specifically as the frames
   allow (read visible titles and labels).
2. `viral_hook_text`: max 10 words, in TRANSCRIPT_LANGUAGE. It MUST mention
   the thing you named in `on_screen` (or the action being done to it: set
   up, connect, compare, fix, type) AND keep the strongest concrete fact of
   the clip: a number, a multiplier, a price, a name ("7x faster", "$136 a
   month", "3,400 stars") from the transcript or the current hook. Never a
   summary of the video's general topic, never a slogan that would fit any
   clip of this video, never drop a figure for a vaguer phrase.
3. `video_title_for_youtube_short`: max 100 chars, same rule, in
   TRANSCRIPT_LANGUAGE, no fake claims.

The current hook and title below were written WITHOUT seeing the frames and
are the kind of topic summary you must replace. Do not reuse their wording.

TRANSCRIPT_LANGUAGE: {language}
CURRENT_HOOK (to replace): {current_hook}
CURRENT_TITLE (to replace): {current_title}
TRANSCRIPT:
{transcript}

Return only:
{{"on_screen": "<one line>", "viral_hook_text": "<max 10 words>", "video_title_for_youtube_short": "<max 100 chars>"}}
"""


class LayoutChoice(BaseModel):
    layout: str
    confidence: float
    why: str


# Scored 94/92/96% over the 48-clip corpus against hand-checked labels, with
# 0-1 false positives out of the 28 clips that must not be touched. Do not
# reword casually: the wins come from the explicit "none is usually right"
# instruction and from naming the exact decorations (corner bugs, score
# counters, subtitles) that four earlier attempts kept mistaking for content.
LAYOUT_CHOICE_PROMPT = """
These frames are sampled at regular intervals from a single landscape video.
You are choosing how to re-frame that video into a vertical 9:16 clip.

Pick ONE layout:

- "none": crop to the speaker and fill the frame. This is the RIGHT answer for
  ordinary talking heads, interviews shot in close-up, b-roll, sport, action,
  music, and any footage whose meaning survives a centre crop. Corner logos,
  score bugs, subscriber counters, lower-thirds and burned-in subtitles do NOT
  change this: they are decoration, and losing them costs nothing.
- "screencast": keep the screen. ONLY when the video is built around a screen
  recording, slides, a spreadsheet, a chart or a map that the viewer must read
  to follow it. If you cannot read words or numbers off the screen that matter
  to the point being made, it is not this.
  (A "camera_inset" option was added here and removed on 31-jul-2026. Whether a
  webcam is composited into a corner of that screen is not something the model
  can see: on the five clips that have one it answered "screencast" every time,
  in both runs, while overall accuracy fell from 92% to 83-85%. camera_inset.py
  finds the same five geometrically with no false positives, so that question is
  answered downstream instead of being asked here.)
- "split": stack two people. ONLY when two people are visible IN THE SAME SHOT
  at the same time in most frames, talking to each other. Frames that alternate
  between one-person close-ups are NOT this, however many people appear.

"none" is by far the most common correct answer. Choose anything else only if
you would defend it to an editor. If you are unsure, answer "none".

confidence is 0..1. why is at most 12 words.
"""


class ShotContentModel(BaseModel):
    shot: int
    kind: str
    focus_left: float
    focus_right: float
    presenter_cam: bool


class ShotContentResponse(BaseModel):
    shots: List[ShotContentModel]


# One still per SHOT of a clip, asked as a closed choice (same lesson as
# LAYOUT_CHOICE_PROMPT: a decision between named options is stable, a free
# measurement is not). It replaced a whole-video prompt that asked for time
# ranges and a width_fraction: that one uploaded the source (a 37-min 1440p
# tutorial is ~670k tokens and a GB-sized upload to get a few numbers back) and
# was never actually called by the pipeline. The focus box is the one number
# still asked for, and it only positions a crop: being 10% off moves the window
# a little, it never changes which layout a shot gets. presenter_cam is the one
# yes/no that gates the per-scene inset (camera_inset.detect_in_scene): without
# it, any small still face or body on a screen was enlarged as the presenter.
SHOT_CONTENT_PROMPT = """
Each image is one shot from the same landscape video, in order, numbered from 0.
The video is being re-framed to a vertical 9:16 clip, and the person who uploaded
it told us it is a screen-based video (a tutorial, a demo, slides).

For EVERY image return one entry with its shot number and a kind:

- "screen": a screen recording, app window, web page, document, spreadsheet,
  slide or code editor fills most of the picture. A webcam bubble or small video
  of the presenter laid over it still makes it "screen". So does a person who
  only appears INSIDE the app (a video being edited, a webcam preview on a page).
- "beside": a chart, slide or screen takes part of the picture and a person
  filmed by a camera stands or sits beside it, each in their own area.
- "camera": camera footage. A person talking to the camera, people, a room,
  b-roll, products, landscapes. Logos, lower-thirds, subtitles or a title card
  over camera footage are still "camera".

focus_left and focus_right (0..1, fractions of the image width) bound the part
of the picture a viewer must READ to follow the shot: the document page, the
web page column, the canvas, the active window or code. Include any webcam
bubble or video of the presenter laid over the screen, and any text paragraph
in full: the box must not cut a face or a line of text. Leave out toolbars, side
panels, browser tabs, docks, desktop wallpaper and empty margins. If the whole
width matters (a full spreadsheet, a full-width slide) answer 0 and 1. For
"camera" answer 0 and 1.

presenter_cam is true only for a "screen" shot with a LIVE webcam window of the
person narrating the video laid over the screen (a bubble, a box, a Loom or
QuickTime window). It is false for everything else, including a person who is
part of the content: a photo or illustration on a slide, a thumbnail or cover
art, a video playing or being edited inside the app, a game character, a
person in a web page. When unsure, answer false. For "beside" and "camera"
answer false.
"""


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message: str) -> None:
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()

SCORE_PROMPT_TEMPLATE = """
You are a senior short-form video strategist.
RANK these candidate windows by how well each would work as a standalone short.

Rules:
- Return only valid JSON.
- Score EVERY window in this batch: exactly one entry per input window, with
  the id you were given. Do not drop the weak ones — say they are weak.
- `score` must be an integer from 0 to 100, and the ranking is what matters:
  use the whole range instead of clustering. Most windows of a normal video
  are not clippable, so reserve 70+ for the ones that pass the test below,
  and put weak filler, housekeeping, outros, rambling transitions and
  low-signal padding under 30 even when the topic is interesting.
- THE 2-SECOND TEST is the main criterion: would the first 2 seconds of this
  moment force a cold viewer (no context) to keep watching? Windows that only
  work with prior context score low.
- Prefer windows with strong hooks, conflict, surprise, outrage, emotion,
  novelty, big numbers, or a clear payoff.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
WINDOWS_JSON:
{windows_json}

Return only:
{{
  "windows": [
    {{
      "id": "<window id>",
      "start": <number>,
      "end": <number>,
      "score": <integer 0-100>,
      "reason": "<very short reason>"
    }}
  ]
}}
"""

DETAIL_PROMPT_TEMPLATE = """
You are a senior short-form video editor and viral copywriter.
Choose the BEST short clips from these shortlisted candidate windows.

CLIP RULES:
- Return only valid JSON.
- Each clip must be {min_secs:g} to {max_secs:g} seconds long, in absolute seconds from the start of the source video.
- Stay within the candidate window boundaries.
- THE 2-SECOND RULE: the clip MUST open on its strongest moment. If the first
  2 seconds would not stop a cold viewer from scrolling, move the start or skip the clip.
- Start slightly before the hook and end slightly after the payoff when possible.
- Do not cut in the middle of a word or phrase.
- No generic intros/outros unless they are the hook.
- STANDS ALONE: the clip must make sense to someone who has seen nothing else.
  If it opens on a pronoun, a "that", a "so anyway", or an answer whose question
  was asked earlier, move the start back to where the idea begins or skip it.
  A brilliant moment that needs the previous five minutes is not a clip.
  Fix this by moving the START earlier, never by cutting the ending short: a
  clip that loses its payoff to gain context has traded down.
- HOW MANY: return {min_clips} to {max_clips} clips. Work through EVERY candidate
  window — they were already scored as the best moments in the video, so a window
  that yields nothing should be the exception, not the norm. Two or three clips
  from one window are fine when they are genuinely different moments. The rules
  above let you skip a weak clip; they are not a licence to return one clip and
  stop. Only fall short of {min_clips} when the material truly does not hold
  them, and never pad with a clip you would not publish yourself.
- DIVERSITY: never return two clips that make the same point, tell the same
  story, or land the same joke — even across different windows. Pick the
  stronger one and drop the other. Two clips on the same broad topic are fine
  as long as each lands its own moment.

HOOK PLAYBOOK — pick the strongest fitting pattern for `viral_hook_text` (max 10 words):
- Open question: "Why does everyone get this wrong?"
- Hot take / controversy: "Stop doing this. Seriously."
- Number / fact shock: "97% of people miss this."
- Story loop: "This one email almost ruined me."
- POV / pattern interrupt: "POV: you finally understand it."
(These are English PATTERNS — always write the actual hook in TRANSCRIPT_LANGUAGE.)
- ABOUT THIS MOMENT, NOT THE VIDEO: the hook and the title name the concrete
  thing that happens inside this clip — the tool being set up, the action,
  the number, the claim, the name. A line that could sit on any clip of this
  video ("I automated my clips with AI") is wrong. If nothing concrete can be
  named, quote the clip's strongest sentence instead of summarising the topic.

COPY RULES — ALL text fields (descriptions, title, hook) MUST be written in TRANSCRIPT_LANGUAGE ({language}):
- Descriptions (TikTok + Instagram): 1-2 punchy sentences that tease the payoff
  without spoiling it, then 3-5 topically relevant hashtags. No generic hashtag spam.
- `video_title_for_youtube_short`: max 100 chars, curiosity-driven, no fake claims.
- `predicted_score`: honest 0-100 estimate of viral potential.
- `why`: one sentence, max 20 words, naming what makes THIS moment worth a
  clip — the specific hook, claim, number or payoff, not the topic.
- `youtube_description_intro`: 2-4 complete sentences, roughly 45-90 words,
  forming a truthful, clip-specific opening paragraph. Describe what happens,
  the key action or payoff, and enough context for a cold viewer to understand
  the moment. Do not pad it with generic phrases or repeat the same point.
  Do not include hashtags, links, channel calls to action, playlists, music
  credits, or facts outside the candidate-window transcript. Only mention a
  game, mode, map, person, hardware, resolution, frame rate, or recording
  setting when the transcript explicitly establishes it.
- `youtube_tags`: 5-15 distinct specific plain tags without `#`, as a JSON
  array. They must be grounded in this clip's transcript, not generic spam.
- `youtube_hashtags`: 3-5 relevant hashtag strings as a JSON array; include
  `#shorts` when appropriate. Do not duplicate these in the intro.
- `thumbnail_brief`: an object that describes the exact selected moment:
  `subject`, `moment`, `emotion`, `visual_style`, `composition`,
  `thumbnail_text` (two to four short words), `must_include` (array), and
  `avoid` (array). Never invent a game, movie, person, episode, result, or
  other fact that is not stated in this candidate-window transcript.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
CANDIDATE_WINDOWS_JSON:
{windows_json}

Return only:
{{
  "shorts": [
    {{
      "start": <number>,
      "end": <number>,
      "source_window_id": "<window id>",
      "predicted_score": <integer 0-100>,
      "video_description_for_tiktok": "<description + hashtags>",
      "video_description_for_instagram": "<description + hashtags>",
      "video_title_for_youtube_short": "<title max 100 chars>",
      "viral_hook_text": "<short overlay max 10 words>",
      "why": "<one sentence, max 20 words>",
      "youtube_description_intro": "<2-4 factual sentences, roughly 45-90 words, describing this clip and its payoff>",
      "youtube_tags": ["<specific plain tag>", "..."],
      "youtube_hashtags": ["#shorts", "..."],
      "thumbnail_brief": {{
        "subject": "<concrete subject>",
        "moment": "<exact moment>",
        "emotion": "<emotion>",
        "visual_style": "<visual style>",
        "composition": "<composition>",
        "thumbnail_text": "<2-4 word overlay>",
        "must_include": ["<visual fact>"],
        "avoid": ["<unrelated visual>"]
      }}
    }}
  ]
}}
"""


LOCAL_WINDOW_DETAIL_PROMPT_TEMPLATE = """
You are a senior short-form video editor and viral copywriter.
Evaluate exactly ONE transcript window. Return either its ONE strongest
standalone short clip, or null when this window has no publishable moment.

CLIP RULES:
- Return only valid JSON.
- The clip must be {min_secs:g} to {max_secs:g} seconds long, in absolute
  seconds from the source video.
- It must stay inside {window_start:g} to {window_end:g} seconds.
- Open on the strongest understandable sentence; do not begin on a dangling
  pronoun, answer, or continuation that needs earlier context.
- End after the payoff. Do not pad a weak moment merely to return a clip.
- `predicted_score` is an honest integer from 0 to 100.
- All text fields must be written in TRANSCRIPT_LANGUAGE ({language}).
- TikTok and Instagram descriptions are 1-2 punchy sentences followed by
  3-5 relevant hashtags.
- The YouTube title is at most 100 characters and makes no unsupported claim.
- `viral_hook_text` is at most 10 words and names this exact moment.
- `why` is one sentence of at most 20 words.
- `youtube_description_intro` is 2-4 factual sentences, roughly 45-90 words,
  explaining the moment and payoff to a cold viewer. Do not include hashtags,
  links, calls to action, or facts absent from this transcript window.
- `youtube_tags` contains 5-15 distinct specific strings without `#`.
- `youtube_hashtags` contains 3-5 relevant hashtag strings.
- `thumbnail_brief` stays grounded in this transcript. Never invent a game,
  movie, person, hardware specification, result, or event.
{correction}

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
WINDOW_ID: {window_id}
WINDOW_START_SECONDS: {window_start}
WINDOW_END_SECONDS: {window_end}
WINDOW_TRANSCRIPT:
{window_text}

Return only:
{{
  "clip": null
}}

or:

{{
  "clip": {{
    "start": <number>,
    "end": <number>,
    "predicted_score": <integer 0-100>,
    "video_description_for_tiktok": "<description + hashtags>",
    "video_description_for_instagram": "<description + hashtags>",
    "video_title_for_youtube_short": "<title max 100 chars>",
    "viral_hook_text": "<short overlay max 10 words>",
    "why": "<one sentence, max 20 words>",
    "youtube_description_intro": "<2-4 factual sentences, roughly 45-90 words>",
    "youtube_tags": ["<specific plain tag>", "..."],
    "youtube_hashtags": ["#shorts", "..."],
    "thumbnail_brief": {{
      "subject": "<concrete subject>",
      "moment": "<exact moment>",
      "emotion": "<emotion>",
      "visual_style": "<visual style>",
      "composition": "<composition>",
      "thumbnail_text": "<2-4 word overlay>",
      "must_include": ["<grounded visual fact>"],
      "avoid": ["<unrelated visual>"]
    }}
  }}
}}
"""


LOCAL_CLIP_RANK_PROMPT_TEMPLATE = """
You are the final editor choosing among independently extracted short clips.
Rank EVERY candidate globally against every other candidate.

Rules:
- Return exactly one entry for every candidate_id supplied.
- Never invent, omit, or duplicate a candidate_id.
- `score` is an integer from 0 to 100 and should use the full range.
- Prefer a strong first two seconds, a self-contained idea, a clear payoff,
  specificity, emotion, surprise, and replay value.
- Penalize clips that need missing context, repeat another candidate's point,
  or end before the payoff.
- Judge the clip transcript and moment, not the writing quality of its title.
- Return only valid JSON.

TRANSCRIPT_LANGUAGE: {language}
CANDIDATES_JSON:
{candidates_json}

Return only:
{{
  "candidates": [
    {{
      "candidate_id": "<supplied candidate id>",
      "score": <integer 0-100>,
      "reason": "<short comparison-based reason>"
    }}
  ]
}}
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _escape_invalid_unicode_escapes(text: str) -> str:
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "u":
            hex_digits = text[i + 2:i + 6]
            if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                chars.append("\\\\u")
                i += 2
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def _parse_json_response_text(text: str) -> dict:
    if not text:
        raise ModelResponseError("Model returned an empty response body.")
    candidate = _extract_json_candidate(text).replace("\x00", "").strip()
    if not candidate:
        raise ModelResponseError("Model response did not contain a JSON object.")
    parse_attempts = [candidate]
    sanitized_candidate = _escape_invalid_unicode_escapes(candidate)
    if sanitized_candidate != candidate:
        parse_attempts.append(sanitized_candidate)
    last_error: Optional[Exception] = None
    for parse_candidate in parse_attempts:
        try:
            return json.loads(parse_candidate)
        except json.JSONDecodeError as e:
            last_error = e
    raise ModelResponseError(f"Failed to parse model JSON response: {last_error}")


def validate_local_response_items(payload, schema, *, allow_empty=False):
    """Validate a local selection response without losing good list items.

    Gemini applies ``response_schema`` server-side, so its whole-response
    validation remains the normal all-or-nothing contract. Local OpenAI-
    compatible models can produce one malformed item beside useful ones. For
    the score/detail/visual-selection schemas, retain each independently valid
    item and let the caller decide whether an empty response is meaningful.
    """
    if schema is LocalWindowDetailResponse:
        if not isinstance(payload, dict):
            raise ModelResponseError("Model response was not a JSON object.")
        raw_clip = payload.get("clip")
        if raw_clip is None:
            return schema.model_validate({"clip": None}).model_dump()
        if not isinstance(raw_clip, dict):
            raise ModelResponseError("Model response 'clip' was not an object or null.")

        # Boundaries are the only irreplaceable local detail output. A small
        # model must not lose a valid cut merely because optional publishing
        # copy has the wrong primitive type.
        clip = dict(raw_clip)
        for key in (
            "source_window_id",
            "video_description_for_tiktok",
            "video_description_for_instagram",
            "video_title_for_youtube_short",
            "viral_hook_text",
            "why",
            "youtube_description_intro",
        ):
            if not isinstance(clip.get(key), str):
                clip[key] = ""
        try:
            score = int(round(float(clip.get("predicted_score", 0))))
        except (TypeError, ValueError, OverflowError):
            score = 0
        clip["predicted_score"] = min(100, max(0, score))
        for key in ("youtube_tags", "youtube_hashtags"):
            value = clip.get(key)
            if isinstance(value, str):
                value = [value]
            clip[key] = [item for item in (value or []) if isinstance(item, str)] \
                if isinstance(value, list) else []
        brief = clip.get("thumbnail_brief")
        if not isinstance(brief, dict):
            brief = {}
        brief = dict(brief)
        for key in (
            "subject",
            "moment",
            "emotion",
            "visual_style",
            "composition",
            "thumbnail_text",
        ):
            if not isinstance(brief.get(key), str):
                brief[key] = ""
        for key in ("must_include", "avoid"):
            value = brief.get(key)
            if isinstance(value, str):
                value = [value]
            brief[key] = [item for item in (value or []) if isinstance(item, str)] \
                if isinstance(value, list) else []
        clip["thumbnail_brief"] = brief
        try:
            return schema.model_validate({"clip": clip}).model_dump()
        except ValidationError as error:
            raise ModelResponseError(
                "Local detail response did not contain valid clip boundaries."
            ) from error

    if schema is ScoreResponse:
        key, item_schema = "windows", ScoredWindowModel
    elif schema is DetailResponse:
        key, item_schema = "shorts", DetailClipModel
    elif schema is VisualResponse:
        key, item_schema = "shorts", VisualClipModel
    elif schema is LocalClipRankResponse:
        key, item_schema = "candidates", LocalRankedCandidateModel
    else:
        # This helper is deliberately narrow: other schemas do not use the
        # local transcript model and must retain their original strict shape.
        return schema.model_validate(payload).model_dump()

    if not isinstance(payload, dict):
        raise ModelResponseError("Model response was not a JSON object.")
    raw_items = payload.get(key)
    if not isinstance(raw_items, list):
        raise ModelResponseError(f"Model response did not contain a list named '{key}'.")
    if not raw_items:
        if allow_empty:
            return schema.model_validate({key: []}).model_dump()
        raise ModelResponseError(f"Model response did not contain any '{key}' items.")

    if schema is not LocalClipRankResponse:
        try:
            return schema.model_validate(payload).model_dump()
        except ValidationError:
            pass

    valid_items = []
    for item in raw_items:
        try:
            # The score reconciliation path deliberately replaces model
            # start/end with the canonical input window and treats ``reason``
            # as display-only. Do not discard an otherwise useful local score
            # merely because a small model omitted one of those fields; an id
            # and score are still required and validated by Pydantic.
            if schema is ScoreResponse and isinstance(item, dict):
                item = {
                    "start": 0,
                    "end": 0,
                    "reason": str(item.get("reason") or ""),
                    "id": item.get("id"),
                    "score": item.get("score"),
                }
            elif schema is LocalClipRankResponse and isinstance(item, dict):
                # Small local models occasionally overshoot the requested
                # 0..100 range. Normalize that arithmetic mistake here so one
                # otherwise useful ranking entry is not discarded.
                item = dict(item)
                score = int(round(float(item.get("score"))))
                item["score"] = min(100, max(0, score))
            valid_items.append(item_schema.model_validate(item).model_dump())
        except (ValidationError, TypeError, ValueError, OverflowError):
            # A single malformed model item must not discard the rest.
            continue

    if not valid_items:
        raise ModelResponseError(f"Model response did not contain any valid '{key}' items.")
    return schema.model_validate({key: valid_items}).model_dump()


def upload_media(client, path, mime_type=None):
    """Upload a local file to the Gemini Files API, by handle and never by path.

    Handed a path, the SDK copies ``os.path.basename(path)`` verbatim into the
    ``X-Goog-Upload-File-Name`` header, and httpx encodes header values as
    ASCII. The downloaded source is named after the video's title, so every
    video whose title is written in Japanese, Cyrillic, Arabic or Greek died
    before a byte left the container with ``'ascii' codec can't encode
    characters in position 0-4`` (prod, 20-sep-2026) — on the three stages
    that upload the whole file: the silent-footage vision pass, the screencast
    detector and the AI editor. Passing an open handle skips that header
    entirely; the readable name still travels as ``display_name``, which goes
    in the JSON body and is UTF-8 all the way.
    """
    guessed = mime_type or mimetypes.guess_type(path)[0] or ""
    # Every caller uploads the source video; an extension the stdlib does not
    # know (or knows as octet-stream) must not reach the API as a type it
    # refuses, so fall back to the container the pipeline always writes.
    if not guessed.startswith(("video/", "audio/", "image/")):
        guessed = "video/mp4"
    with open(path, "rb") as fh:
        return client.files.upload(
            file=fh,
            config={"mime_type": guessed,
                    "display_name": os.path.basename(path)},
        )


class GeminiBlockedError(ValueError):
    """The API refused the request for content-policy reasons.

    Deterministic: the same payload is rejected every time (verified in prod,
    23-jul-2026 — a stand-up video came back PROHIBITED_CONTENT in ~300ms on
    every attempt), and BLOCK_NONE safety settings do NOT lift it. Retrying is
    pointless, so callers must fail fast with a message that tells the user the
    video's content is the problem, not the service."""


_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST",
                           "SPII", "IMAGE_SAFETY", "RECITATION"}


def raise_if_blocked(response):
    """Raise GeminiBlockedError when the API refused to answer on policy grounds."""
    pf = getattr(response, "prompt_feedback", None)
    reason = getattr(pf, "block_reason", None)
    if reason:
        name = getattr(reason, "name", None) or str(reason)
        raise GeminiBlockedError(
            f"Gemini blocked this video's content ({name}). The AI provider's "
            "usage policies reject this material, so it can't be analyzed.")
    for c in (getattr(response, "candidates", None) or []):
        fr = getattr(c, "finish_reason", None)
        name = (getattr(fr, "name", None) or str(fr or "")).upper()
        if name in _BLOCKED_FINISH_REASONS:
            raise GeminiBlockedError(
                f"Gemini blocked its answer for this video ({name}). The AI "
                "provider's usage policies reject this material, so it can't be analyzed.")


def _get_response_text(response) -> str:
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def _calculate_cost_analysis(response, model_name: str) -> Optional[dict]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    prices = lookup_model_prices(model_name)
    price_estimated = prices is None
    if prices is None:
        # Unknown model: conservative estimate so the UI shows something sane.
        prices = (0.50, 3.00)
    input_price_per_million, output_price_per_million = prices
    prompt_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    # Thinking tokens bill at the output rate even though they are invisible.
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = ((output_tokens + thinking_tokens) / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name,
        "price_estimated": price_estimated,
    }


def _thinking_config_from_env(model_name: str):
    """GEMINI_THINKING_SCORE: off (default) | low | high | <token budget>.

    Applied only to the scoring stage. Gemini 3 models take thinking_level,
    Gemini 2.5 takes thinking_budget; returns None (= model default) if the
    setting is off or the SDK rejects the config."""
    raw = (os.getenv("GEMINI_THINKING_SCORE") or "off").strip().lower()
    if raw in ("", "off", "0", "none", "false"):
        return None
    try:
        if raw.isdigit():
            return genai_types.ThinkingConfig(thinking_budget=int(raw))
        if raw in ("low", "high"):
            if model_name.startswith("gemini-3"):
                return genai_types.ThinkingConfig(thinking_level=raw)
            return genai_types.ThinkingConfig(thinking_budget=2048 if raw == "low" else 8192)
    except Exception as e:
        _log(f"⚠️ Ignoring GEMINI_THINKING_SCORE={raw!r}: {e}")
    return None


def _config_for_strategy(strategy: str, mode: str, model_name: str) -> genai_types.GenerateContentConfig:
    # The detail stage writes creative copy (hooks/descriptions) — it gets a
    # high temperature; timestamps are validated and word-snapped afterwards.
    # The score stage stays precise. Fallback strategies get conservative.
    creative = mode == "detail"
    kwargs = {
        "response_mime_type": "application/json",
        "candidate_count": 1,
    }
    if strategy == "strict-json":
        kwargs["temperature"] = 0.7 if creative else 0.1
    elif strategy == "json-text-recovery":
        kwargs["temperature"] = 0.2 if creative else 0.0
    else:  # structured-schema: schema-enforced output, primary strategy
        kwargs["temperature"] = 0.9 if creative else 0.2
        kwargs["response_schema"] = DetailResponse if mode == "detail" else ScoreResponse
        if mode == "score":
            thinking = _thinking_config_from_env(model_name)
            if thinking is not None:
                kwargs["thinking_config"] = thinking
    return genai_types.GenerateContentConfig(**kwargs)


def main() -> int:
    _configure_stdio()

    parser = argparse.ArgumentParser(description="Run a single Gemini request for clip scoring/detailing.")
    parser.add_argument("--mode", choices=["score", "detail"], required=True)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--strategy", default="structured-schema")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY.")

    with open(args.input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = args.model
    client = genai.Client(api_key=api_key)
    config = _config_for_strategy(args.strategy, args.mode, model_name)
    language = str(payload.get("language") or "unknown")

    template = SCORE_PROMPT_TEMPLATE if args.mode == "score" else DETAIL_PROMPT_TEMPLATE
    fmt = {
        "video_duration": payload["video_duration"],
        "language": language,
        "windows_json": json.dumps(payload["windows"], ensure_ascii=False),
    }
    if args.mode != "score":
        # Score mode receives every window, not a shortlist, so a count target
        # derived from it would be meaningless — and the score template has no
        # placeholder for one: it ranks whatever it is given.
        fmt["min_clips"], fmt["max_clips"] = clip_count_targets(len(payload.get("windows") or []))
        fmt["min_secs"], fmt["max_secs"] = clip_duration_bounds()
    prompt = template.format(**fmt)

    _log(f"🤖 Gemini worker request: mode={args.mode} strategy={args.strategy} model={model_name} items={len(payload.get('windows', []))}")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    raw_text = _get_response_text(response)
    # With response_schema the SDK returns an already-validated object; fall
    # back to the text-repair path only when that is unavailable.
    parsed_obj = getattr(response, "parsed", None)
    if parsed_obj is not None:
        parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
    else:
        parsed = _parse_json_response_text(raw_text)
    result = {
        "mode": args.mode,
        "payload": parsed,
        "cost_analysis": _calculate_cost_analysis(response, model_name),
        "raw_text": raw_text,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"✅ Gemini worker success: mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
