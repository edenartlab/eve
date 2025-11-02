import json
import re
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
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_length = len(response.content)
    assert content_length > 0, "Content length is 0"
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
    if not markdown_text:
        return []

    # Media extensions
    media_exts = r'jpg|jpeg|png|gif|webp|bmp|svg|mp4|webm|mov|mp3|wav'

    # Match markdown images/media: ![alt](url) or [text](url)
    # This pattern extracts only the URL from inside the parentheses
    markdown_pattern = rf'!?\[.*?\]\((https?://[^\)]+\.(?:{media_exts}))\)'

    # Match HTML tags: <img>, <video>, <audio>, <source>
    html_pattern = rf'<(?:img|video|audio|source)[^>]+src=["\']([^"\']+\.(?:{media_exts}))["\']'

    # Match plain URLs with media extensions (but exclude those inside brackets)
    # Exclude ] and ( to avoid matching URLs in markdown alt text
    plain_pattern = rf'(?<!\[)(https?://[^\s<>"\'\)\]\[]+\.(?:{media_exts}))(?!\])'

    urls = []
    urls.extend(re.findall(markdown_pattern, markdown_text, re.IGNORECASE))
    urls.extend(re.findall(html_pattern, markdown_text, re.IGNORECASE))

    # For plain URLs, first remove all markdown image syntax to avoid duplicates
    text_without_markdown = re.sub(markdown_pattern, '', markdown_text, flags=re.IGNORECASE)
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
    """Validate a single media URL (image, video, or audio).

    Args:
        url: The media URL to validate

    Raises:
        ValueError: If the media is invalid or cannot be validated
    """
    media_type = get_media_type(url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Validate based on media type
        if media_type == 'image':
            ok, info = validate_image_bytes(response.content)
            if not ok:
                raise ValueError(f"Invalid image from {url}: {info.get('reason', 'unknown')}")

        elif media_type == 'video':
            ok, info = validate_video_bytes(response.content)
            if not ok:
                raise ValueError(f"Invalid video from {url}: {info.get('reason', 'unknown')}")

        elif media_type == 'audio':
            ok, info = validate_audio_bytes(response.content)
            if not ok:
                raise ValueError(f"Invalid audio from {url}: {info.get('reason', 'unknown')}")

        else:
            raise ValueError(f"Unknown media type for {url}")

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate {media_type} from post: {url} - {e}")


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
    for url in media_urls:
        validate_media_url(url)


def validate_eden_session_id(data: Dict[str, Any]) -> None:
    """Validate that eden_session_id is a valid BSON ObjectId."""

    if "eden_session_id" not in data:
        raise ValueError("Missing 'eden_session_id' field in JSON")

    try:
        Session.from_mongo(data["eden_session_id"])
    except (InvalidId, TypeError) as e:
        raise ValueError(f"Invalid eden_session_id: {e}")


def validate_creation(title: str, tagline: str, poster_image: str, post: str, session_id: str) -> None:
    """Validate a creation with all its media assets.

    Args:
        title: Creation title
        tagline: Creation tagline
        poster_image: URL to the poster image
        post: Markdown post content
        session_id: Eden session ID
    """
    try:
        # Validate poster image
        if not poster_image:
            raise ValueError("Missing poster_image")

        if not poster_image.startswith(("http://", "https://")):
            raise ValueError(f"Invalid poster_image URL: {poster_image}")

        response = requests.get(poster_image, timeout=30)
        response.raise_for_status()

        ok, info = validate_image_bytes(response.content)
        if not ok:
            raise ValueError(f"Invalid poster image {poster_image}: {info.get('reason', 'unknown')}")

        # Validate media in post
        media_urls = extract_media_urls(post)

        for idx, url in enumerate(media_urls, 1):
            validate_media_url(url)

        # Validate session ID
        try:
            Session.from_mongo(session_id)
        except (InvalidId, TypeError) as e:
            raise ValueError(f"Invalid session_id: {e}")

        # check blog post at least 10 chars
        if len(post) < 20:
            raise Exception("Blog post must be at least 20 characters long")

        logger.info(f"Creation validated successfully!")

    except Exception as e:
        logger.error(f"✗✗✗ VALIDATION FAILED ✗✗✗\nError: {e}")
        raise


def validate_ipfs_bundle(ipfs_hash: str) -> None:
    """Main validation function for IPFS bundle."""

    try:
        content = download_from_ipfs(ipfs_hash)
        data = validate_json(content)

        validate_image_field(data)
        validate_post_images(data)
        validate_eden_session_id(data)

        logger.info(f"IPFS bundle validated: {ipfs_hash}")

    except Exception as e:
        logger.error(f"✗✗✗ VALIDATION FAILED ✗✗✗\nError: {e}")
        raise


if __name__ == "__main__":
    validate_ipfs_bundle("QmTmeCCrx4WeHNfTpUB1jHLpoukYirw1ydS2hjxfEfDreN")
    #validate_creation("Test Title", "Test Tagline", "https://d14i3advvh2bvd.cloudfront.net/58656ca3bf3013df15536f8cba11dbe224a353d866e5e5bbef070a377fb5bc36.jpeg", "Test Post containing an image: ![Test Image](https://d14i3advvh2bvd.cloudfront.net/cffc6e0704676f7ea9c2325943796df8a2a7ee56e9ed47e63e82c53b5da22e18.jpg), ![Test Image 2](https://d14i3advvh2bvd.cloudfront.net/48f4b354bd5711b4d2234a9b8f05b193e186df8639ca72273c068e8b4f1910f1.png)", "690201918de005f84abc8163")