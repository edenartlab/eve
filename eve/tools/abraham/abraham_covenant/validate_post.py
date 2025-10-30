import json
import re
import sys
import requests
from typing import Dict, List, Any
from loguru import logger
from bson.errors import InvalidId

from eve.agent.session.models import Session
from eve.utils.file_utils import (
    validate_image_bytes,
    validate_video_bytes,
    validate_audio_bytes,
)


def download_from_ipfs(ipfs_hash: str) -> bytes:
    """Download content from IPFS gateway."""
    url = f"https://ipfs.io/ipfs/{ipfs_hash}"
    logger.info(f"Downloading from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_length = len(response.content)
    logger.info(f"Downloaded {content_length} bytes from IPFS")
    return response.content


def validate_json(content: bytes) -> Dict[str, Any]:
    """Validate that content is valid JSON and return parsed data."""
    try:
        data = json.loads(content.decode('utf-8'))
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
        raise ValueError(f"Invalid image {source}: {info.get('reason', 'unknown')}")


def get_media_type(url: str) -> str:
    """Determine media type from URL extensions."""
    url_lower = url.lower()

    if any(url_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']):
        return 'image'
    elif any(url_lower.endswith(ext) for ext in ['.mp4', '.webm', '.mov']):
        return 'video'
    elif any(url_lower.endswith(ext) for ext in ['.mp3', '.wav']):
        return 'audio'
    else:
        return 'unknown'


def extract_media_urls(markdown_text: str) -> List[str]:
    """Extract image, video, and audio URLs from markdown text."""
    # Media extensions
    media_exts = r'jpg|jpeg|png|gif|webp|bmp|svg|mp4|webm|mov|mp3|wav'

    # Match markdown images/media: ![alt](url) or [text](url)
    markdown_pattern = rf'!?\[.*?\]\((https?://[^\)]+\.(?:{media_exts}))\)'

    # Match HTML tags: <img>, <video>, <audio>, <source>
    html_pattern = rf'<(?:img|video|audio|source)[^>]+src=["\']([^"\']+\.(?:{media_exts}))["\']'

    # Match plain URLs with media extensions
    plain_pattern = rf'(https?://[^\s<>"\'\)]+\.(?:{media_exts}))'

    urls = []
    urls.extend(re.findall(markdown_pattern, markdown_text, re.IGNORECASE))
    urls.extend(re.findall(html_pattern, markdown_text, re.IGNORECASE))
    urls.extend(re.findall(plain_pattern, markdown_text, re.IGNORECASE))

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def validate_post_images(data: Dict[str, Any]) -> None:
    """Validate all media (images, videos, audio) referenced in the post field."""

    if "post" not in data:
        raise ValueError("Missing 'post' field in JSON")

    post_content = data["post"]
    if not isinstance(post_content, str):
        raise ValueError(f"Post field is not a string, got {type(post_content)}")

    # Extract media URLs
    media_urls = extract_media_urls(post_content)

    if not media_urls:
        return

    # Validate each media file
    for idx, url in enumerate(media_urls, 1):
        media_type = get_media_type(url)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Validate based on media type
            if media_type == 'image':
                ok, info = validate_image_bytes(response.content)
                if not ok:
                    raise ValueError(f"Invalid image from {url}: {info.get('reason', 'unknown')}")
                width, height = info["size"]

            elif media_type == 'video':
                ok, info = validate_video_bytes(response.content)
                if not ok:
                    raise ValueError(f"Invalid video from {url}: {info.get('reason', 'unknown')}")
                width, height = info["size"]
                duration = info.get("duration", 0)

            elif media_type == 'audio':
                ok, info = validate_audio_bytes(response.content)
                if not ok:
                    raise ValueError(f"Invalid audio from {url}: {info.get('reason', 'unknown')}")
                duration = info.get("duration", 0)
                sample_rate = info.get("sample_rate", 0)

            else:
                raise ValueError(f"Unknown media type for {url}")

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to validate {media_type} from post: {url} - {e}")


def validate_eden_session_id(data: Dict[str, Any]) -> None:
    """Validate that eden_session_id is a valid BSON ObjectId."""

    if "eden_session_id" not in data:
        raise ValueError("Missing 'eden_session_id' field in JSON")

    try:
        session = Session.from_mongo(data["eden_session_id"])
    except (InvalidId, TypeError) as e:
        raise ValueError(f"Invalid eden_session_id: {e}")


def validate_ipfs_bundle(ipfs_hash: str) -> None:
    """Main validation function for IPFS bundle."""

    try:
        content = download_from_ipfs(ipfs_hash)
        data = validate_json(content)

        validate_image_field(data)
        validate_post_images(data)
        validate_eden_session_id(data)
        
        logger.success(f"IPFS bundle validated: {ipfs_hash}")

    except Exception as e:
        logger.error(f"✗✗✗ VALIDATION FAILED ✗✗✗\nError: {e}")
        raise


if __name__ == "__main__":
    ipfs_hash = sys.argv[1]
    validate_ipfs_bundle(ipfs_hash)
