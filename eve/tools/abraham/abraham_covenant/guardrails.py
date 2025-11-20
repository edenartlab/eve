import sys
from typing import Any, Dict

from loguru import logger

from eve.tools.abraham.guardrails import (
    download_from_ipfs,
    download_with_retry,
    extract_media_urls,
    validate_eden_session_id,
    validate_image_field,
    validate_json,
    validate_media_url,
    validate_session_id_string,
)
from eve.utils.file_utils import validate_image_bytes, validate_video_bytes


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


def validate_creation(
    title: str, tagline: str, poster_image: str, post: str, video: str, session_id: str
) -> None:
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

        content = download_with_retry(poster_image)
        ok, info = validate_image_bytes(content)
        if not ok:
            raise ValueError(
                f"Invalid poster image {poster_image}: {info.get('reason', 'unknown')}"
            )

        # Validate media in post
        media_urls = extract_media_urls(post)

        for idx, url in enumerate(media_urls, 1):
            validate_media_url(url)

        # Validate video
        if not video:
            raise ValueError("Missing video")

        if not video.startswith(("http://", "https://")):
            raise ValueError(f"Invalid video URL: {video}")

        content = download_with_retry(video)
        ok, info = validate_video_bytes(content)
        if not ok:
            raise ValueError(f"Invalid video {video}: {info.get('reason', 'unknown')}")

        # Validate session ID
        validate_session_id_string(session_id)

        # check blog post at least 10 chars
        if len(post) < 20:
            raise Exception("Blog post must be at least 20 characters long")

        logger.info("Creation validated successfully!")

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
    ipfs_hash = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "QmT8PHxuXxE7xc711ohYYVSDiqs6G1NTKSFjPiEEyBUJKv"
    )
    validate_ipfs_bundle(ipfs_hash)
