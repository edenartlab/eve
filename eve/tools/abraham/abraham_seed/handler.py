from eve.mongo import Collection, Document
from eve.tool import ToolContext
from bson import ObjectId
from typing import Literal, Optional
from pydantic import BaseModel
from datetime import datetime


class AbrahamCreation(BaseModel):
    index: int
    title: str
    tagline: str
    poster_image: str
    blog_post: str
    video: Optional[str] = None
    session_id: str
    contract_address: str
    tx_hash: str
    ipfs_hash: str
    explorer_url: str
    minted_at: datetime


@Collection("abraham_seeds")
class AbrahamSeed(Document):
    session_id: ObjectId
    title: str
    proposal: str
    tagline: str
    cast_hash: str
    image: str
    url: str
    status: Literal["seed", "creation", "archived"]
    creation: Optional[AbrahamCreation] = None


async def handler(context: ToolContext):
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
    if not context.session:
        raise ValueError("Session ID is required")

    title = context.args.get("title")
    proposal = context.args.get("proposal")
    tagline = context.args.get("tagline")
    cast_hash = context.args.get("cast_hash")
    image = context.args.get("image")

    # Validate required fields
    if not all([title, proposal, tagline, cast_hash, image]):
        raise ValueError(
            "All parameters are required: title, proposal, tagline, cast_hash, image"
        )

    # Generate URL
    url = f"https://abraham.ai/seeds/{context.session}"

    seed = AbrahamSeed(
        session_id=ObjectId(context.session),
        title=title,
        proposal=proposal,
        tagline=tagline,
        cast_hash=cast_hash,
        image=image,
        url=url,
        status="seed",
    )
    seed.save()

    return {"output": {"url": url, "seed_id": str(seed.id)}}
