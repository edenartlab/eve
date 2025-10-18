import os
import logging
import asyncio
from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_seed.handler import AbrahamSeed, AbrahamCreation

from eve.utils.chain_utils import (
    safe_send,
    BlockchainError,
    load_contract,
    Network,
)
from eve.utils.ipfs_utils import pin as ipfs_pin

# Initialize logger
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CONTRACT_ADDRESS_COVENANT = os.getenv("CONTRACT_ADDRESS_COVENANT")
ABRAHAM_PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")

# ABI file is stored locally in the tool folder
CONTRACT_ABI_COVENANT = os.path.join(
    os.path.dirname(__file__),
    "abi_covenant.json"
)


def commit_daily_work(
    index: int,
    title: str,
    tagline: str,
    poster_image: str,
    blog_post: str,
    session_id: str
):
    """
    Commit Abraham's daily work to the blockchain.

    Args:
        index: Work index number
        title: Title of the work
        tagline: Short description/tagline
        poster_image: URL to the poster image
        blog_post: Full blog post content
        session_id: Eden session ID
    """
    try:
        if not ABRAHAM_PRIVATE_KEY:
            raise BlockchainError("ABRAHAM_PRIVATE_KEY not configured")
        if not CONTRACT_ADDRESS_COVENANT:
            raise BlockchainError("CONTRACT_ADDRESS_COVENANT not configured")

        w3, owner, contract, abi = load_contract(
            address=CONTRACT_ADDRESS_COVENANT,
            abi_path=CONTRACT_ABI_COVENANT,
            private_key=ABRAHAM_PRIVATE_KEY,
            network=Network.ETH_SEPOLIA
        )

        # Upload poster image to IPFS
        logger.info(f"Uploading poster image to IPFS: {poster_image}")
        image_cid = ipfs_pin(poster_image)
        poster_image_hash = image_cid.split("/")[-1]

        # Create metadata JSON
        json_data = {
            "name": title,
            "description": tagline,
            "post": blog_post,
            "external_url": f"https://abraham.ai/creation/{index}",
            "eden_session_id": session_id,
            "image": f"ipfs://{poster_image_hash}",
            "attributes": [
                # {"trait_type": "Artist", "value": "Abraham"},
            ],
        }

        logger.info(f"Metadata: {json_data}")

        # Upload metadata to IPFS
        ipfs_hash = ipfs_pin(json_data)
        logger.info(f"Metadata uploaded to IPFS: {ipfs_hash}")

        # Prepare contract function call
        contract_function = contract.functions.commitDailyWork(
            f"ipfs://{ipfs_hash}"
        )

        # Send transaction
        tx_hash, receipt = safe_send(
            w3,
            contract_function,
            ABRAHAM_PRIVATE_KEY,
            op_name="ABRAHAM_DAILY_WORK",
            nonce=None,
            value=0,
            abi=abi,
            network=Network.ETH_SEPOLIA,
        )

        # Build explorer URL for ETH Sepolia
        tx_hash_hex = tx_hash.hex()
        if not tx_hash_hex.startswith('0x'):
            tx_hash_hex = f"0x{tx_hash_hex}"
        explorer_url = f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"

        logger.info(f"✅ Daily work committed successfully: {tx_hash_hex}")
        logger.info(f"Explorer: {explorer_url}")

        return {
            "tx_hash": tx_hash_hex,
            "ipfs_hash": ipfs_hash,
            "image_hash": poster_image_hash,
            "explorer_url": explorer_url
        }

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_DAILY_WORK failed: {e}")
        raise


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")
    if not session:
        raise Exception("Session is required")

    title = args.get("title")
    tagline = args.get("tagline")
    poster_image = args.get("poster_image")
    blog_post = args.get("post")

    abraham_seed = AbrahamSeed.find_one({
        "session_id": ObjectId(session)
    })

    logger.info("Processing Abraham creation")
    logger.info(f"Title: {title}")
    logger.info(f"Tagline: {tagline}")
    logger.info(f"Post: {blog_post}")
    logger.info(f"Poster image: {poster_image}")

    # Commit to blockchain
    try:
        result = commit_daily_work(
            index=1,  # Using index 1 as specified
            title=title,
            tagline=tagline,
            poster_image=poster_image,
            blog_post=blog_post,
            session_id=session
        )

        # Update creation status
        abraham_seed.update(
            status="creation",
            creation=AbrahamCreation(
                index=1,
                title=title,
                tagline=tagline,
                poster_image=poster_image,
                blog_post=blog_post,

                tx_hash=result["tx_hash"],
                ipfs_hash=result["ipfs_hash"],
                explorer_url=result["explorer_url"]
            ).model_dump()
        )

        return {
            "output": [{
                "session": session,
                "tx_hash": result["tx_hash"],
                "ipfs_hash": result["ipfs_hash"],
                "explorer_url": result["explorer_url"]
            }]
        }
    except Exception as e:
        logger.error(f"Failed to commit daily work: {e}")
        raise

