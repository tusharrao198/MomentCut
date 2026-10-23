"""Small, strict client for a local Postiz instance.

This module supports the one-click OpenShorts -> Postiz -> YouTube handoff.
It intentionally accepts only the private Docker-network endpoint used by the
local runbook.  The configured endpoint receives both the rendered MP4 and the
Postiz API key, so treating it as an arbitrary URL would create an SSRF and
credential-exfiltration path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any, Dict, List
from urllib.parse import quote

import httpx

from clip_packaging import normalise_tags


EXPECTED_API_BASE_URL = "http://postiz:5000/api/public/v1"
DEFAULT_TIMEOUT_SECONDS = 300.0
MIN_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 900.0
POST_TYPES = frozenset(("draft", "now"))
PRIVACY_TYPES = frozenset(("private", "unlisted", "public"))
THUMBNAIL_MIME_TYPES = frozenset(("image/jpeg", "image/png"))
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024


class PostizError(RuntimeError):
    """A definite Postiz/configuration error that is safe to show to the user."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class PostizOutcomeUnknown(PostizError):
    """The create-post request may have reached Postiz; never blindly retry."""


@dataclass(frozen=True)
class PostizConfig:
    api_base_url: str
    api_key: str
    youtube_integration_id: str
    post_type: str
    privacy_status: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "PostizConfig":
        api_base_url = (os.environ.get("POSTIZ_API_BASE_URL") or "").strip()
        if api_base_url != EXPECTED_API_BASE_URL:
            raise PostizError(
                "Postiz is not configured for the private Docker network. "
                f"Set POSTIZ_API_BASE_URL={EXPECTED_API_BASE_URL}."
            )

        api_key = (os.environ.get("POSTIZ_API_KEY") or "").strip()
        if not api_key:
            raise PostizError("Missing POSTIZ_API_KEY. Create a Postiz Public API key first.")

        post_type = (os.environ.get("POSTIZ_DEFAULT_POST_TYPE") or "draft").strip().lower()
        if post_type not in POST_TYPES:
            raise PostizError("POSTIZ_DEFAULT_POST_TYPE must be either 'draft' or 'now'.")

        privacy_status = (
            os.environ.get("POSTIZ_YOUTUBE_PRIVACY_STATUS") or "private"
        ).strip().lower()
        if privacy_status not in PRIVACY_TYPES:
            raise PostizError(
                "POSTIZ_YOUTUBE_PRIVACY_STATUS must be private, unlisted, or public."
            )

        raw_timeout = (os.environ.get("POSTIZ_TIMEOUT_SECONDS") or "").strip()
        if not raw_timeout:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        else:
            try:
                timeout_seconds = float(raw_timeout)
            except ValueError as exc:
                raise PostizError(
                    f"POSTIZ_TIMEOUT_SECONDS must be {int(MIN_TIMEOUT_SECONDS)}–"
                    f"{int(MAX_TIMEOUT_SECONDS)} seconds."
                ) from exc
        if not MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise PostizError(
                f"POSTIZ_TIMEOUT_SECONDS must be {int(MIN_TIMEOUT_SECONDS)}–"
                f"{int(MAX_TIMEOUT_SECONDS)} seconds."
            )

        return cls(
            api_base_url=api_base_url,
            api_key=api_key,
            youtube_integration_id=(
                os.environ.get("POSTIZ_YOUTUBE_INTEGRATION_ID") or ""
            ).strip(),
            post_type=post_type,
            privacy_status=privacy_status,
            timeout_seconds=timeout_seconds,
        )


def configuration() -> PostizConfig:
    """Return valid local Postiz configuration, never logging the API key."""
    return PostizConfig.from_env()


def is_configured() -> bool:
    try:
        configuration()
        return True
    except PostizError:
        return False


def configured_post_type() -> str | None:
    try:
        return configuration().post_type
    except PostizError:
        return None


def _headers(config: PostizConfig) -> Dict[str, str]:
    # Postiz expects the raw API key. Prefixing this value with "Bearer " makes
    # a valid key fail authentication.
    return {"Authorization": config.api_key}


def _client(config: PostizConfig, **kwargs) -> httpx.Client:
    timeout = httpx.Timeout(config.timeout_seconds, connect=min(10.0, config.timeout_seconds))
    return httpx.Client(timeout=timeout, follow_redirects=False, **kwargs)


def _endpoint(config: PostizConfig, suffix: str) -> str:
    return f"{config.api_base_url}/{suffix.lstrip('/')}"


def _error_text(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
        if isinstance(body, dict):
            value = body.get("message") or body.get("error") or body.get("detail")
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
    except (ValueError, TypeError):
        pass
    return response.text.strip().replace("\n", " ")[:300] or "Postiz returned an empty error."


def _raise_definite_http_error(response: httpx.Response, action: str) -> None:
    raise PostizError(
        f"Postiz {action} failed ({response.status_code}): {_error_text(response)}",
        status_code=response.status_code,
    )


def _json(response: httpx.Response, action: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise PostizError(f"Postiz {action} returned invalid JSON.") from exc


def _list_from_response(data: Any) -> List[dict]:
    """Support the current array response and the documented wrapper variants."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("integrations", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise PostizError("Postiz integrations returned an unexpected response.")


def list_integrations(config: PostizConfig) -> List[dict]:
    try:
        with _client(config) as client:
            response = client.get(_endpoint(config, "integrations"), headers=_headers(config))
    except httpx.RequestError as exc:
        raise PostizError(f"Could not reach local Postiz: {type(exc).__name__}.") from exc
    if response.status_code >= 400 or response.is_redirect:
        _raise_definite_http_error(response, "integration lookup")
    return _list_from_response(_json(response, "integration lookup"))


def resolve_youtube_integration(config: PostizConfig) -> dict:
    integrations = list_integrations(config)
    enabled_youtube = [
        item for item in integrations
        if (
            item.get("identifier") == "youtube"
            and item.get("disabled") is not True
            and isinstance(item.get("id"), str)
            and item["id"].strip()
        )
    ]
    wanted = config.youtube_integration_id
    if wanted:
        for item in enabled_youtube:
            if item["id"].strip() == wanted:
                return {**item, "id": item["id"].strip()}
        raise PostizError(
            "POSTIZ_YOUTUBE_INTEGRATION_ID is not an enabled YouTube integration."
        )
    if len(enabled_youtube) != 1:
        if not enabled_youtube:
            raise PostizError("No enabled YouTube integration found in Postiz.")
        raise PostizError(
            "More than one enabled YouTube integration exists. "
            "Set POSTIZ_YOUTUBE_INTEGRATION_ID in .env."
        )
    return {**enabled_youtube[0], "id": enabled_youtube[0]["id"].strip()}


def _uploaded_media(data: Any) -> dict:
    if isinstance(data, dict):
        candidate = data.get("data") if isinstance(data.get("data"), dict) else data
        media_id = candidate.get("id")
        path = candidate.get("path")
        if (
            isinstance(media_id, str)
            and media_id.strip()
            and isinstance(path, str)
            and path.strip()
        ):
            return {"id": media_id.strip(), "path": path.strip()}
    raise PostizError("Postiz upload returned no usable media id and path.")


def upload_file(config: PostizConfig, file_path: str | Path, filename: str) -> dict:
    """Stream one already-verified local MP4 to Postiz; never read it all at once."""
    try:
        with open(file_path, "rb") as media, _client(config) as client:
            response = client.post(
                _endpoint(config, "upload"),
                headers=_headers(config),
                files={"file": (filename, media, "video/mp4")},
            )
    except OSError as exc:
        raise PostizError(f"Could not read rendered clip: {type(exc).__name__}.") from exc
    except httpx.RequestError as exc:
        # The upload cannot yet have been used to create a YouTube post; retry
        # is safe, although it may leave an unused Postiz media record.
        raise PostizError(f"Postiz media upload failed: {type(exc).__name__}.") from exc
    if response.status_code >= 400 or response.is_redirect:
        _raise_definite_http_error(response, "media upload")
    return _uploaded_media(_json(response, "media upload"))


def upload_thumbnail(
    config: PostizConfig,
    content: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """Upload one validated JPEG/PNG thumbnail to Postiz."""
    if content_type not in THUMBNAIL_MIME_TYPES:
        raise PostizError("Thumbnail must be a JPEG or PNG image.")
    if not content:
        raise PostizError("Thumbnail image is empty.")
    if len(content) > MAX_THUMBNAIL_BYTES:
        raise PostizError("Thumbnail image exceeds the 2 MB limit.")

    try:
        with BytesIO(content) as thumbnail, _client(config) as client:
            response = client.post(
                _endpoint(config, "upload"),
                headers=_headers(config),
                files={"file": (filename, thumbnail, content_type)},
            )
    except httpx.RequestError as exc:
        # This upload alone cannot create a YouTube post. Retrying is safe,
        # though Postiz may retain an unused media record.
        raise PostizError(f"Postiz thumbnail upload failed: {type(exc).__name__}.") from exc
    if response.status_code >= 400 or response.is_redirect:
        _raise_definite_http_error(response, "thumbnail upload")
    return _uploaded_media(_json(response, "thumbnail upload"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_youtube_post(
    config: PostizConfig,
    integration_id: str,
    media: dict,
    title: str,
    description: str,
    thumbnail: dict | None = None,
    *,
    tags: List[str] | None = None,
) -> str:
    """Create one Postiz YouTube draft/now post.

    Any ambiguous failure is raised as ``PostizOutcomeUnknown``. The caller
    records that state and must not retry automatically, or it could create a
    duplicate YouTube upload.
    """
    integration_id = integration_id.strip() if isinstance(integration_id, str) else ""
    media_id = media.get("id") if isinstance(media, dict) else None
    media_path = media.get("path") if isinstance(media, dict) else None
    media_id = media_id.strip() if isinstance(media_id, str) else ""
    media_path = media_path.strip() if isinstance(media_path, str) else ""
    if not integration_id:
        raise PostizError("Postiz returned an invalid YouTube integration id.")
    if not media_id or not media_path:
        raise PostizError("Saved Postiz media state is invalid.")
    youtube_tags = [
        # Postiz validates both display text and canonical value. They are the
        # same for OpenShorts' already-normalised plain-string tags.
        {"label": label, "value": label}
        for label in normalise_tags(tags or [], maximum_items=100)
    ]

    thumbnail_settings = {}
    if thumbnail is not None:
        thumbnail_id = thumbnail.get("id") if isinstance(thumbnail, dict) else None
        thumbnail_path = thumbnail.get("path") if isinstance(thumbnail, dict) else None
        thumbnail_id = thumbnail_id.strip() if isinstance(thumbnail_id, str) else ""
        thumbnail_path = thumbnail_path.strip() if isinstance(thumbnail_path, str) else ""
        if not thumbnail_id or not thumbnail_path:
            raise PostizError("Saved Postiz thumbnail state is invalid.")
        thumbnail_settings = {
            "thumbnail": {
                "id": thumbnail_id,
                "path": thumbnail_path,
            }
        }

    payload = {
        "type": config.post_type,
        "date": _iso_now(),
        "shortLink": False,
        "tags": [],
        "posts": [{
            "integration": {"id": integration_id},
            "value": [{
                "content": description,
                "image": [{"id": media_id, "path": media_path}],
            }],
            "settings": {
                "__type": "youtube",
                "title": title,
                "type": config.privacy_status,
                "selfDeclaredMadeForKids": "no",
                "tags": youtube_tags,
                **thumbnail_settings,
            },
        }],
    }
    try:
        with _client(config) as client:
            response = client.post(
                _endpoint(config, "posts"), headers=_headers(config), json=payload
            )
    except httpx.RequestError as exc:
        raise PostizOutcomeUnknown(
            "Postiz did not confirm post creation. Check Postiz before trying again."
        ) from exc

    if response.is_redirect or response.status_code >= 500:
        raise PostizOutcomeUnknown(
            "Postiz did not confirm post creation. Check Postiz before trying again.",
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        _raise_definite_http_error(response, "post creation")

    try:
        data = response.json()
    except ValueError as exc:
        raise PostizOutcomeUnknown(
            "Postiz did not confirm post creation. Check Postiz before trying again."
        ) from exc
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise PostizOutcomeUnknown(
            "Postiz did not confirm post creation. Check Postiz before trying again."
        )
    post_id = data[0].get("postId")
    if not isinstance(post_id, str) or not post_id:
        raise PostizOutcomeUnknown(
            "Postiz did not confirm post creation. Check Postiz before trying again."
        )
    return post_id


def update_youtube_thumbnail(
    config: PostizConfig,
    post_id: str,
    thumbnail: dict,
) -> None:
    """Update the thumbnail on one existing unpublished Postiz YouTube post.

    This is deliberately separate from ``create_youtube_post``: updating
    settings is idempotent, whereas retrying post creation can duplicate a
    YouTube upload.
    """
    post_id = post_id.strip() if isinstance(post_id, str) else ""
    if not post_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", post_id):
        raise PostizError("Saved Postiz draft id is invalid.")
    thumbnail_id = thumbnail.get("id") if isinstance(thumbnail, dict) else None
    thumbnail_path = thumbnail.get("path") if isinstance(thumbnail, dict) else None
    thumbnail_id = thumbnail_id.strip() if isinstance(thumbnail_id, str) else ""
    thumbnail_path = thumbnail_path.strip() if isinstance(thumbnail_path, str) else ""
    if not thumbnail_id or not thumbnail_path:
        raise PostizError("Saved Postiz thumbnail state is invalid.")

    try:
        with _client(config) as client:
            response = client.put(
                _endpoint(config, f"posts/{quote(post_id, safe='')}/settings"),
                headers=_headers(config),
                json={
                    "settings": {
                        "thumbnail": {"id": thumbnail_id, "path": thumbnail_path},
                    },
                },
            )
    except httpx.RequestError as exc:
        # This exact settings update is safe to retry. It cannot create a
        # second post or a second YouTube upload.
        raise PostizError(
            "Postiz did not confirm the thumbnail update. Try updating the draft again."
        ) from exc
    if response.status_code >= 400 or response.is_redirect:
        _raise_definite_http_error(response, "thumbnail update")


def preflight() -> dict:
    """Authenticated, non-secret check used by the Windows runbook."""
    config = configuration()
    integration = resolve_youtube_integration(config)
    return {
        "ok": True,
        "postType": config.post_type,
        "youtubeIntegrationId": str(integration.get("id") or ""),
        "youtubeIntegrationName": str(integration.get("name") or "YouTube"),
    }
