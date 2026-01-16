"""
Generic Seed model for session-to-blockchain registration.
Seeds create an immutable IPFS + on-chain record tied to a session ID.
"""

from datetime import datetime
from typing import Literal, Optional

from bson import ObjectId
from pydantic import BaseModel

from eve.mongo import Collection, Document


class SeedMetadata(BaseModel):
    """Blockchain registration metadata for a minted seed."""

    contract_address: str
    tx_hash: str
    ipfs_hash: str
    image_hash: Optional[str] = None
    explorer_url: str
    minted_at: datetime


@Collection("seeds")
class Seed(Document):
    """
    Generic seed document representing an on-chain session registration.

    Seeds are created automatically on first user message for opted-in agents,
    or explicitly via tools like abraham_seed.
    """

    session_id: ObjectId
    agent_id: ObjectId
    title: str
    tagline: Optional[str] = None
    proposal: Optional[str] = None
    image: Optional[str] = None
    seed_metadata: Optional[SeedMetadata] = None
    status: Literal["pending", "minted", "failed"] = "pending"
    platform: Optional[str] = None
    is_dm: bool = False


"""

Message (ipfs hash)
- eden_message_id : string
- sender : address
- content : string
- attachments : ipfsHash[]

Session (Seed)
- eden_session_id : string
- messages : Message[]
- reactions : Dict[address, string]
- title : string
- abstract : string
- poster_image : ipfsHash

Creation
- session : Session
- title : string
- abstract : string
- poster_image : ipfsHash
- video : ipfsHash
- article : string

"""
