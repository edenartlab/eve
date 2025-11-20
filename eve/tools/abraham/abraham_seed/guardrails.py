import json
from typing import Any, Dict

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
from eve.utils.file_utils import validate_image_bytes


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


def validate_eden_session_id(data: Dict[str, Any]) -> None:
    """Validate that eden_session_id is a valid BSON ObjectId."""

    if "eden_session_id" not in data:
        raise ValueError("Missing 'eden_session_id' field in JSON")

    try:
        Session.from_mongo(data["eden_session_id"])
    except (InvalidId, TypeError) as e:
        raise ValueError(f"Invalid eden_session_id: {e}")


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


def validate_seed(
    title: str, tagline: str, proposal: str, image: str, session_id: str
) -> None:
    """Validate a seed with its image asset.

    Args:
        title: Seed title
        tagline: Seed tagline
        proposal: Creation proposal
        image: URL to the representative/main image
        session_id: Eden session ID
    """
    try:
        # Validate image
        if not image:
            raise ValueError("Missing image")

        if not image.startswith(("http://", "https://")):
            raise ValueError(f"Invalid image URL: {image}")

        content = download_with_retry(image)
        ok, info = validate_image_bytes(content)
        if not ok:
            raise ValueError(f"Invalid image {image}: {info.get('reason', 'unknown')}")

        # Validate session ID
        try:
            Session.from_mongo(session_id)
        except (InvalidId, TypeError) as e:
            raise ValueError(f"Invalid session_id: {e}")

        # Check minimum content lengths
        if len(title) < 3:
            raise ValueError("Title must be at least 3 characters long")

        if len(tagline) < 10:
            raise ValueError("Tagline must be at least 10 characters long")

        if len(proposal) < 20:
            raise ValueError("Proposal must be at least 20 characters long")

        logger.info("Seed validated successfully!")

    except Exception as e:
        logger.error(f"✗✗✗ SEED VALIDATION FAILED ✗✗✗\nError: {e}")
        raise


def validate_ipfs_bundle(ipfs_hash: str) -> None:
    """Main validation function for IPFS seed bundle."""

    try:
        content = download_from_ipfs(ipfs_hash)
        data = validate_json(content)

        # Validate required fields
        required_fields = ["title", "description", "proposal", "image"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in JSON")

        validate_image_field(data)
        validate_eden_session_id(data)

        logger.info(f"IPFS seed bundle validated: {ipfs_hash}")

    except Exception as e:
        logger.error(f"✗✗✗ VALIDATION FAILED ✗✗✗\nError: {e}")
        raise
