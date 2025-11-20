from loguru import logger

from eve.tools.abraham.guardrails import (
    download_from_ipfs,
    download_with_retry,
    validate_eden_session_id,
    validate_image_field,
    validate_json,
    validate_session_id_string,
)
from eve.utils.file_utils import validate_image_bytes


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
        validate_session_id_string(session_id)

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
