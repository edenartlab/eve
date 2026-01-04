"""
VerdelisSeed - A snapshot of an idea for Verdelis.

Seeds are simple containers representing an initial idea that will later
be expanded into a full Storyboard. They contain a title, logline,
contributing agents, and exemplary images that represent the concept.
"""

import asyncio
import logging
from typing import List

import requests
from pydantic import Field

from eve.agent import Agent
from eve.mongo import Collection, Document
from eve.tool import ToolContext
from eve.utils.file_utils import validate_image_bytes

logger = logging.getLogger(__name__)


@Collection("verdelis_seeds")
class VerdelisSeed(Document):
    """
    A snapshot of an idea for Verdelis.

    Seeds are simple containers representing an initial creative concept
    that will later be expanded into a full Storyboard. They capture the
    essence of an idea with a title, logline, and 2-4 representative images.

    Attributes:
        title: Title of the seed/idea
        logline: Short logline summarizing the concept
        agents: List of agents that contributed to this seed
        images: Exemplary images that represent the idea (max 2-4)
    """

    title: str
    logline: str
    agents: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        # Convert agents list
        # if "agents" in data:
        #     data["agents"] = [
        #         ObjectId(agent) if isinstance(agent, str) else agent
        #         for agent in data.get("agents", [])
        #     ]
        super().__init__(**data)


def download_url(url: str, timeout: int = 30) -> bytes:
    """Download content from URL and return bytes."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def validate_image_url(url: str) -> tuple[bool, dict]:
    """
    Download and validate that a URL points to a valid image.

    Returns:
        tuple: (ok: bool, info: dict)
    """
    try:
        content = download_url(url)
        return validate_image_bytes(content)
    except requests.RequestException as e:
        return False, {"reason": f"Failed to download: {e}"}
    except Exception as e:
        return False, {"reason": f"Validation error: {e}"}


async def handler(context: ToolContext):
    """
    Create a Verdelis seed - a snapshot of an idea.

    Args:
        args: Dictionary containing:
            - title: Title of the seed
            - logline: Short logline summarizing the concept
            - agents: Array of agent usernames that contributed
            - images: Array of image URLs representing the idea
    """
    title = context.args.get("title")
    logline = context.args.get("logline")
    agents = context.args.get("agents", [])
    images = context.args.get("images", [])

    # Validate required fields
    if not title:
        raise ValueError("Parameter 'title' is required")
    if not logline:
        raise ValueError("Parameter 'logline' is required")
    if not images:
        raise ValueError("Parameter 'images' is required (at least one image)")

    # Validate images is a list of strings
    if not isinstance(images, list):
        raise ValueError("Parameter 'images' must be an array of image URLs")

    for i, img in enumerate(images):
        if not isinstance(img, str):
            raise ValueError(f"images[{i}] must be a string URL")

    logger.info("Creating Verdelis seed...")
    logger.info(f"Title: {title}")
    logger.info(f"Logline: {logline}")
    logger.info(f"Agents: {agents}")
    logger.info(f"Images: {len(images)}")

    # Validate all image URLs are actual images
    invalid_images = []
    for i, image_url in enumerate(images):
        logger.info(f"Validating image {i + 1}/{len(images)}: {image_url}")
        ok, info = await asyncio.to_thread(validate_image_url, image_url)
        if not ok:
            invalid_images.append((i, image_url, info.get("reason", "Unknown error")))
        else:
            logger.info(f"  Valid image: {info.get('format')} {info.get('size')}")

    if invalid_images:
        error_details = "; ".join(
            [f"image {i}: {reason}" for i, url, reason in invalid_images]
        )
        raise ValueError(f"Invalid images: {error_details}")

    # Convert agents to ObjectIds
    try:
        agents = [Agent.load(agent) for agent in agents]
    except Exception as e:
        raise ValueError(f"Invalid agents: {e}")

    agent_usernames = [agent.username for agent in agents]

    # Create the seed
    seed = VerdelisSeed(
        title=title,
        logline=logline,
        agents=agent_usernames,
        images=images,
    )
    seed.save()

    logger.info(f"Seed created successfully: {seed.id}")

    return {
        "output": {
            "artifact_id": str(seed.id),
            "title": title,
            "image_count": len(images),
        }
    }
