import logging
import os
from datetime import datetime
from typing import Literal, Optional

from bson import ObjectId
from pydantic import BaseModel
from web3 import Web3

from eve.mongo import Collection, Document
from eve.tool import ToolContext
from eve.tools.abraham.abraham_seed.guardrails import (
    validate_ipfs_bundle,
    validate_seed,
)
from eve.utils.chain_utils import (
    BlockchainError,
    Network,
    load_contract,
    safe_send,
)
from eve.utils.ipfs_utils import pin as ipfs_pin

# Initialize logger
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CONTRACT_ADDRESS_SEEDS = os.getenv("CONTRACT_ADDRESS_SEEDS")
ABRAHAM_PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")

# ABI file is stored locally in the tool folder
CONTRACT_ABI_SEEDS = os.path.join(os.path.dirname(__file__), "abi_seeds.json")


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


class SeedMetadata(BaseModel):
    contract_address: str
    tx_hash: str
    ipfs_hash: str
    image_hash: str
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
    seed_metadata: Optional[SeedMetadata] = None
    creation: Optional[AbrahamCreation] = None


def mint_seed(
    title: str,
    tagline: str,
    proposal: str,
    image: str,
    session_id: str,
):
    """
    Mint Abraham's seed to the blockchain.

    Args:
        title: Title of the seed
        tagline: Short description/tagline
        proposal: Creation proposal
        image: URL to the representative/main image
        session_id: Eden session ID
    """
    try:
        if not ABRAHAM_PRIVATE_KEY:
            raise BlockchainError("ABRAHAM_PRIVATE_KEY not configured")
        if not CONTRACT_ADDRESS_SEEDS:
            raise BlockchainError("CONTRACT_ADDRESS_SEEDS not configured")

        # Convert address to checksum format
        contract_address = Web3.to_checksum_address(CONTRACT_ADDRESS_SEEDS)

        # Upload image to IPFS
        logger.info(f"Uploading image to IPFS: {image}")
        image_cid = ipfs_pin(image)
        image_hash = image_cid.split("/")[-1]

        # Create metadata JSON
        json_data = {
            "title": title,
            "description": tagline,
            "proposal": proposal,
            "image": f"ipfs://{image_hash}",
            "eden_session_id": session_id,
            "attributes": [],
        }

        logger.info(f"Seed metadata: {json_data}")

        # Upload metadata to IPFS
        ipfs_hash = ipfs_pin(json_data)
        logger.info(f"Metadata uploaded to IPFS: {ipfs_hash}")

        # Validate IPFS bundle
        validate_ipfs_bundle(ipfs_hash)
        logger.info(f"IPFS seed bundle is valid: {ipfs_hash}")

        # Prepare contract function call
        w3, owner, contract, abi = load_contract(
            address=contract_address,
            abi_path=CONTRACT_ABI_SEEDS,
            private_key=ABRAHAM_PRIVATE_KEY,
            network=Network.BASE_SEPOLIA,
        )

        contract_function = contract.functions.submitSeed(f"ipfs://{ipfs_hash}")

        # Send transaction
        tx_hex, _ = safe_send(
            w3,
            contract_function,
            ABRAHAM_PRIVATE_KEY,
            op_name="ABRAHAM_SUBMIT_SEED",
            nonce=None,
            value=0,
            abi=abi,
            network=Network.BASE_SEPOLIA,
        )

        # Build explorer URL
        if not tx_hex.startswith("0x"):
            tx_hex = f"0x{tx_hex}"

        explorer_url = f"https://sepolia.basescan.org/tx/{tx_hex}"

        logger.info(f"✅ Seed submitted successfully: {tx_hex}")
        logger.info(f"Explorer: {explorer_url}")

        return {
            "tx_hash": tx_hex,
            "ipfs_hash": ipfs_hash,
            "image_hash": image_hash,
            "explorer_url": explorer_url,
            "contract_address": contract_address,
        }

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_SUBMIT_SEED failed: {e}")
        raise


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
        session: Eden session ID
    """
    if not context.session:
        raise ValueError("Session ID is required")

    title = context.args.get("title")
    proposal = context.args.get("proposal")
    tagline = context.args.get("tagline")
    cast_hash = context.args.get("cast_hash")
    image = context.args.get("image")

    # Validate required fields individually
    if not title:
        raise ValueError("Parameter 'title' is required")
    if not proposal:
        raise ValueError("Parameter 'proposal' is required")
    if not tagline:
        raise ValueError("Parameter 'tagline' is required")
    if not cast_hash:
        raise ValueError("Parameter 'cast_hash' is required")
    if not image:
        raise ValueError("Parameter 'image' is required")

    session_id = str(context.session)

    # Safety checks
    validate_seed(title, tagline, proposal, image, session_id)

    logger.info("Seed validated successfully!")
    logger.info(f"Title: {title}")
    logger.info(f"Tagline: {tagline}")
    logger.info(f"Proposal: {proposal}")
    logger.info(f"Image: {image}")

    # Mint to blockchain before saving to DB
    try:
        result = mint_seed(
            title=title,
            tagline=tagline,
            proposal=proposal,
            image=image,
            session_id=session_id,
        )

        # Generate URL
        url = f"https://abraham.ai/seeds/{context.session}"

        seed_metadata = SeedMetadata(
            contract_address=result["contract_address"],
            tx_hash=result["tx_hash"],
            ipfs_hash=result["ipfs_hash"],
            image_hash=result["image_hash"],
            explorer_url=result["explorer_url"],
            minted_at=datetime.now(),
        )

        seed = AbrahamSeed(
            session_id=ObjectId(context.session),
            title=title,
            proposal=proposal,
            tagline=tagline,
            cast_hash=cast_hash,
            image=image,
            url=url,
            status="seed",
            seed_metadata=seed_metadata.model_dump(),
        )
        seed.save()

        return {
            "output": {
                "url": url,
                "seed_id": str(seed.id),
                "tx_hash": result["tx_hash"],
                "ipfs_hash": result["ipfs_hash"],
                "explorer_url": result["explorer_url"],
            }
        }

    except Exception as e:
        logger.error(f"Failed to mint seed: {e}")
        raise
