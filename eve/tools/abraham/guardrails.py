"""Shared guardrails functionality for Abraham seed and covenant validation."""

import json
import re
from typing import Any, Dict, List

import requests
from bson.errors import InvalidId
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eve.agent.session.models import Session
from eve.utils.file_utils import (
    validate_audio_bytes,
    validate_image_bytes,
    validate_video_bytes,
)


class IPFSDownloadError(Exception):
    """Error during IPFS download operations."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, IPFSDownloadError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying download (attempt {retry_state.attempt_number}/3)..."
    ),
)
def _download_from_gateway(url: str, timeout: int = 60) -> bytes:
    """Download content from a single gateway with retry logic."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_length = len(response.content)
        if content_length == 0:
            raise IPFSDownloadError("Content length is 0")
        return response.content
    except requests.exceptions.HTTPError as e:
        # Non-gateway errors (404, 403, etc) should not be retried
        if e.response.status_code not in [502, 503, 504]:
            raise IPFSDownloadError(f"HTTP error {e.response.status_code}: {e}")
        raise  # Gateway errors (502, 503, 504) will be retried by tenacity
    except Exception as e:
        raise IPFSDownloadError(f"Download failed: {e}")


def download_from_ipfs(ipfs_hash: str) -> bytes:
    """Download content from IPFS gateway with automatic fallback to multiple gateways"""
    last_exception = None
    for gateway in [
        "https://ipfs.io/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
        "https://dweb.link/ipfs/",
        "https://gateway.pinata.cloud/ipfs/",
    ]:
        url = f"{gateway}{ipfs_hash}"
        try:
            return _download_from_gateway(url)
        except Exception as e:
            last_exception = e
            continue
    error_msg = f"Failed to download IPFS content {ipfs_hash} from all gateways"
    raise IPFSDownloadError(f"{error_msg}. Last error: {last_exception}")


def validate_json(content: bytes) -> Dict[str, Any]:
    """Validate that content is valid JSON and return parsed data."""
    try:
        data = json.loads(content.decode("utf-8"))
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Content is not valid JSON: {e}")


def validate_image_field(data: Dict[str, Any]) -> None:
    """Validate the image field in the bundle."""

    if "image" not in data:
        raise ValueError("Missing 'image' field in JSON")

    image_value = data.get("image")
    if not image_value.startswith("ipfs://"):
        raise ValueError(f"Image field does not start with 'ipfs://': {image_value}")

    ipfs_hash = image_value.replace("ipfs://", "")
    image_bytes = download_from_ipfs(ipfs_hash)
    ok, info = validate_image_bytes(image_bytes)
    if not ok:
        raise ValueError(f"Invalid image: {info.get('reason', 'unknown')}")


def validate_session_id_string(session_id: str) -> None:
    """Validate that session_id is a valid BSON ObjectId.

    Args:
        session_id: Session ID string to validate

    Raises:
        ValueError: If session_id is invalid
    """
    try:
        Session.from_mongo(session_id)
    except (InvalidId, TypeError) as e:
        raise ValueError(f"Invalid session_id: {e}")


def validate_eden_session_id(data: Dict[str, Any]) -> None:
    """Validate that eden_session_id field in data dict is a valid BSON ObjectId.

    Args:
        data: Dictionary containing 'eden_session_id' field

    Raises:
        ValueError: If 'eden_session_id' field is missing or invalid
    """
    if "eden_session_id" not in data:
        raise ValueError("Missing 'eden_session_id' field in JSON")

    validate_session_id_string(data["eden_session_id"])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying download (attempt {retry_state.attempt_number}/3)..."
    ),
)
def download_with_retry(url: str, timeout: int = 60) -> bytes:
    """Download content from URL with retry logic."""
    response = requests.get(url, timeout=timeout)
    # Non-gateway errors should not be retried
    if response.status_code >= 400:
        if response.status_code not in [502, 503, 504]:
            response.raise_for_status()
    response.raise_for_status()
    content_length = len(response.content)
    if content_length == 0:
        raise requests.RequestException("Content length is 0")
    return response.content


def get_media_type(url: str) -> str:
    """Determine media type from URL extensions."""
    url_lower = url.lower()

    if any(
        url_lower.endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"]
    ):
        return "image"
    elif any(url_lower.endswith(ext) for ext in [".mp4", ".webm", ".mov"]):
        return "video"
    elif any(url_lower.endswith(ext) for ext in [".mp3", ".wav"]):
        return "audio"
    else:
        return "unknown"


def extract_media_urls(markdown_text: str) -> List[str]:
    """Extract image, video, and audio URLs from markdown text."""
    if not markdown_text:
        return []

    # Media extensions
    media_exts = r"jpg|jpeg|png|gif|webp|bmp|svg|mp4|webm|mov|mp3|wav"

    # Match markdown images/media: ![alt](url) or [text](url)
    # This pattern extracts only the URL from inside the parentheses
    markdown_pattern = rf"!?\[.*?\]\((https?://[^\)]+\.(?:{media_exts}))\)"

    # Match HTML tags: <img>, <video>, <audio>, <source>
    html_pattern = (
        rf'<(?:img|video|audio|source)[^>]+src=["\']([^"\']+\.(?:{media_exts}))["\']'
    )

    # Match plain URLs with media extensions (but exclude those inside brackets)
    # Exclude ] and ( to avoid matching URLs in markdown alt text
    plain_pattern = rf'(?<!\[)(https?://[^\s<>"\'\)\]\[]+\.(?:{media_exts}))(?!\])'

    urls = []
    urls.extend(re.findall(markdown_pattern, markdown_text, re.IGNORECASE))
    urls.extend(re.findall(html_pattern, markdown_text, re.IGNORECASE))

    # For plain URLs, first remove all markdown image syntax to avoid duplicates
    text_without_markdown = re.sub(
        markdown_pattern, "", markdown_text, flags=re.IGNORECASE
    )
    urls.extend(re.findall(plain_pattern, text_without_markdown, re.IGNORECASE))

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def validate_media_url(url: str) -> None:
    """Validate a single media URL (image, video, or audio)."""
    media_type = get_media_type(url)
    try:
        content = download_with_retry(url)

        # Validate based on media type
        if media_type == "image":
            ok, info = validate_image_bytes(content)
            if not ok:
                raise ValueError(
                    f"Invalid image from {url}: {info.get('reason', 'unknown')}"
                )

        elif media_type == "video":
            ok, info = validate_video_bytes(content)
            if not ok:
                raise ValueError(
                    f"Invalid video from {url}: {info.get('reason', 'unknown')}"
                )

        elif media_type == "audio":
            ok, info = validate_audio_bytes(content)
            if not ok:
                raise ValueError(
                    f"Invalid audio from {url}: {info.get('reason', 'unknown')}"
                )

        else:
            raise ValueError(f"Unknown media type for {url}")

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate {media_type} from post: {url} - {e}")
