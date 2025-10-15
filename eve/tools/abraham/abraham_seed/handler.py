from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal, Optional
from pydantic import BaseModel


class AbrahamCreation(BaseModel):
    index: int
    title: str
    tagline: str
    poster_image: str
    blog_post: str
    tx_hash: str
    ipfs_hash: str
    explorer_url: str


@Collection("abraham_seeds")
class AbrahamSeed(Document):
    session_id: ObjectId
    title: str
    proposal: str
    tagline: str
    cast_hash: str
    image: str
    url: str
    status: Literal["seed", "creation"]
    creation: Optional[AbrahamCreation] = None


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    """
    Save an Abraham seed after a creation has been made and cast.

    Args:
        args: Dictionary containing:
            - title: Title of the creation
            - proposal: Creation proposal
            - tagline: A short tagline for the seed
            - cast_hash: The cast hash from the farcaster_cast tool
            - image: A representative/main image URL from the creation
        session: The session ID where the creation was made (provided automatically)
    """
    if not session:
        raise ValueError("Session ID is required")

    title = args.get("title")
    proposal = args.get("proposal")
    tagline = args.get("tagline")
    cast_hash = args.get("cast_hash")
    image = args.get("image")

    # Validate required fields
    if not all([title, proposal, tagline, cast_hash, image]):
        raise ValueError("All parameters are required: title, proposal, tagline, cast_hash, image")

    # Generate URL
    url = f"https://abraham.ai/seeds/{session}"

    seed = AbrahamSeed(
        session_id=ObjectId(session),
        title=title,
        proposal=proposal,
        tagline=tagline,
        cast_hash=cast_hash,
        image=image,
        url=url,
        status="seed"
    )
    seed.save()

    return {
        "output": {"url": url, "seed_id": str(seed.id)}
    }
