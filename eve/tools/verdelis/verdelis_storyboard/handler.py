"""
VerdelisStoryboard - A storyboard container for Verdelis movies.

Storyboards contain a title, logline, plot summary, cast of agents,
and an ordered list of image frames that tell the visual story.
Optionally includes music and vocals audio tracks.

Storyboards are expanded from Seeds, and must reference a valid seed.
"""

import asyncio
import logging
from typing import List, Optional

import requests
from bson import ObjectId
from pydantic import Field

from eve.agent import Agent
from eve.mongo import Collection, Document
from eve.tool import ToolContext
from eve.tools.verdelis.verdelis_seed.handler import VerdelisSeed
from eve.utils.file_utils import validate_audio_bytes, validate_image_bytes

logger = logging.getLogger(__name__)


@Collection("verdelis_storyboards")
class VerdelisStoryboard(Document):
    """
    A storyboard for a Verdelis movie.

    Storyboards are expanded from Seeds and contain the full visual story
    with image frames, plot summary, and optional audio tracks.

    Attributes:
        artifact_id: Reference to the VerdelisSeed this storyboard expands
        title: Title of the storyboard
        logline: Short logline summarizing the story
        plot: Plot summary
        agents: Cast of all agent usernames that appear in the storyboard
        session_id: The Eden session ID where this storyboard is drafted
        image_frames: Ordered list of image URLs representing the storyboard frames
        music: Optional URL to background music audio track
        vocals: Optional URL to vocals/narration audio track
    """

    artifact_id: ObjectId
    title: str
    logline: str
    plot: str
    agents: List[str] = Field(default_factory=list)
    session_id: ObjectId
    image_frames: List[str] = Field(default_factory=list)
    music: Optional[str] = None
    vocals: Optional[str] = None

    def __init__(self, **data):
        if isinstance(data.get("artifact_id"), str):
            data["artifact_id"] = ObjectId(data["artifact_id"])
        if isinstance(data.get("session_id"), str):
            data["session_id"] = ObjectId(data["session_id"])
        # data["agents"] = [
        #     ObjectId(agent) if isinstance(agent, str) else agent
        #     for agent in data.get("agents", [])
        # ]
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


def validate_audio_url(url: str) -> tuple[bool, dict]:
    """
    Download and validate that a URL points to a valid audio file.

    Returns:
        tuple: (ok: bool, info: dict)
    """
    try:
        content = download_url(url)
        return validate_audio_bytes(content)
    except requests.RequestException as e:
        return False, {"reason": f"Failed to download: {e}"}
    except Exception as e:
        return False, {"reason": f"Validation error: {e}"}


async def handler(context: ToolContext):
    """
    Create a Verdelis storyboard - validates fields and saves to database.

    Args:
        args: Dictionary containing:
            - artifact_id: ID of the VerdelisSeed this storyboard expands (required)
            - title: Title of the storyboard
            - logline: Short logline summarizing the story
            - plot: Plot summary
            - agents: Array of agent usernames in the cast
            - image_frames: Array of image URLs for storyboard frames
            - music: Optional URL to background music audio track
            - vocals: Optional URL to vocals/narration audio track
        session: Eden session ID
    """
    if not context.session:
        raise ValueError("Session ID is required")

    artifact_id = context.args.get("artifact_id")
    title = context.args.get("title")
    logline = context.args.get("logline")
    plot = context.args.get("plot")
    agents = context.args.get("agents", [])
    image_frames = context.args.get("image_frames", [])
    music = context.args.get("music")
    vocals = context.args.get("vocals")

    # Validate required fields
    if not artifact_id:
        raise ValueError("Parameter 'artifact_id' is required")
    if not title:
        raise ValueError("Parameter 'title' is required")
    if not logline:
        raise ValueError("Parameter 'logline' is required")
    if not plot:
        raise ValueError("Parameter 'plot' is required")
    if not image_frames:
        raise ValueError("Parameter 'image_frames' is required (at least one frame)")

    # Validate seed exists
    try:
        artifact_oid = ObjectId(artifact_id)
    except Exception:
        raise ValueError(f"Invalid artifact ID format: {artifact_id}")

    existing_seed = VerdelisSeed.find_one({"_id": artifact_oid})
    if not existing_seed:
        raise ValueError(f"Seed not found: {artifact_id}")

    logger.info(f"Found seed: {existing_seed.title}")

    # Validate image_frames is a list of strings
    if not isinstance(image_frames, list):
        raise ValueError("Parameter 'image_frames' must be an array of image URLs")

    for i, frame in enumerate(image_frames):
        if not isinstance(frame, str):
            raise ValueError(f"image_frames[{i}] must be a string URL")

    # Verify agents are valid
    try:
        agents = [Agent.from_mongo(a) for a in agents]
    except Exception as e:
        raise ValueError(f"Invalid agents: {e}")

    agent_usernames = [agent.username for agent in agents]

    session_id = str(context.session)

    logger.info("Creating Verdelis storyboard...")
    logger.info(f"Seed: {artifact_id}")
    logger.info(f"Title: {title}")
    logger.info(f"Logline: {logline}")
    logger.info(f"Plot length: {len(plot)} chars")
    logger.info(f"Agents: {agent_usernames}")
    logger.info(f"Image frames: {len(image_frames)}")
    logger.info(f"Music: {music}")
    logger.info(f"Vocals: {vocals}")

    # Validate all image URLs are actual images
    invalid_frames = []
    for i, frame_url in enumerate(image_frames):
        logger.info(f"Validating image frame {i + 1}/{len(image_frames)}: {frame_url}")
        ok, info = await asyncio.to_thread(validate_image_url, frame_url)
        if not ok:
            invalid_frames.append((i, frame_url, info.get("reason", "Unknown error")))
        else:
            logger.info(f"  Valid image: {info.get('format')} {info.get('size')}")

    if invalid_frames:
        error_details = "; ".join(
            [f"frame {i}: {reason}" for i, url, reason in invalid_frames]
        )
        raise ValueError(f"Invalid images: {error_details}")

    # Validate music URL if provided
    if music:
        logger.info(f"Validating music URL: {music}")
        ok, info = await asyncio.to_thread(validate_audio_url, music)
        if not ok:
            raise ValueError(
                f"Invalid music audio: {info.get('reason', 'Unknown error')}"
            )
        logger.info(
            f"  Valid audio: {info.get('format')} duration={info.get('duration')}s"
        )

    # Validate vocals URL if provided
    if vocals:
        logger.info(f"Validating vocals URL: {vocals}")
        ok, info = await asyncio.to_thread(validate_audio_url, vocals)
        if not ok:
            raise ValueError(
                f"Invalid vocals audio: {info.get('reason', 'Unknown error')}"
            )
        logger.info(
            f"  Valid audio: {info.get('format')} duration={info.get('duration')}s"
        )

    # Create the storyboard
    storyboard = VerdelisStoryboard(
        artifact_id=artifact_oid,
        title=title,
        logline=logline,
        plot=plot,
        agents=agent_usernames,
        session_id=ObjectId(session_id),
        image_frames=image_frames,
        music=music,
        vocals=vocals,
    )
    storyboard.save()

    logger.info(f"Storyboard created successfully: {storyboard.id}")

    return {
        "output": {
            "artifact_id": str(storyboard.id),
            "title": title,
            "frame_count": len(image_frames),
        }
    }
