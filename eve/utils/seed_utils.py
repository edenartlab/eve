"""
Seed service for automatic session-to-blockchain registration.
Handles IPFS upload and on-chain seed creation for opted-in agents.
"""

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from bson import ObjectId
from web3 import Web3

from eve.agent.session.seed import Seed, SeedMetadata
from eve.utils.chain_utils import (
    BlockchainError,
    Network,
    load_contract,
    safe_send,
)
from eve.utils.ipfs_utils import pin as ipfs_pin

if TYPE_CHECKING:
    from eve.agent import Agent
    from eve.agent.session.models import Session

logger = logging.getLogger(__name__)

# Get configuration from environment variables
CONTRACT_ADDRESS_SEEDS = os.getenv("CONTRACT_ADDRESS_SEEDS")
ABRAHAM_PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")

# ABI file path (same as abraham_seed)
CONTRACT_ABI_SEEDS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tools",
    "abraham",
    "abraham_seed",
    "abi_seeds.json",
)


def is_agent_seed_enabled(agent: "Agent") -> bool:
    """
    Check if an agent should auto-create seeds on first user message.
    Currently hardcoded for abraham only.
    """
    return agent.username == "abraham"


def is_dm_session(session: "Session") -> bool:
    """
    Detect if a session is a DM (direct message) session.
    Uses session_key patterns to determine DM status.
    """
    session_key = session.session_key or ""

    # Discord DM pattern: discord-dm-{agent_id}-{user_id}
    if "-dm-" in session_key:
        return True

    # Add other platform DM patterns as needed
    return False


def _check_contract_access(w3, contract, owner_address: str) -> None:
    """Pre-flight check to verify contract state and access before minting."""
    try:
        is_paused = contract.functions.paused().call()
        if is_paused:
            raise BlockchainError("Contract is paused - cannot submit seeds")

        creator_role = contract.functions.CREATOR_ROLE().call()
        has_role = contract.functions.hasRole(creator_role, owner_address).call()
        if not has_role:
            raise BlockchainError(
                f"Wallet {owner_address} does not have CREATOR_ROLE on the contract. "
                f"An admin needs to call addCreator('{owner_address}') on the contract."
            )

        try:
            seed_count = contract.functions.seedCount().call()
            max_seeds = contract.functions.MAX_TOTAL_SEEDS().call()
            if seed_count >= max_seeds:
                raise BlockchainError(
                    f"Max total seeds reached ({seed_count}/{max_seeds})"
                )
        except Exception:
            pass  # These checks are optional

        logger.info(
            f"Pre-flight checks passed: paused={is_paused}, has_creator_role={has_role}"
        )

    except BlockchainError:
        raise
    except Exception as e:
        logger.warning(f"Pre-flight check failed (non-critical): {e}")


def mint_seed(
    title: str,
    tagline: str,
    proposal: str,
    session_id: str,
    image: Optional[str] = None,
) -> dict:
    """
    Mint a seed to the blockchain.

    Args:
        title: Title of the seed
        tagline: Short description/tagline
        proposal: Creation proposal
        session_id: Eden session ID
        image: Optional URL to the representative image
    """
    if not ABRAHAM_PRIVATE_KEY:
        raise BlockchainError("ABRAHAM_PRIVATE_KEY not configured")
    if not CONTRACT_ADDRESS_SEEDS:
        raise BlockchainError("CONTRACT_ADDRESS_SEEDS not configured")

    contract_address = Web3.to_checksum_address(CONTRACT_ADDRESS_SEEDS)

    # Upload image to IPFS if provided
    image_hash = None
    if image:
        logger.info(f"Uploading image to IPFS: {image}")
        image_cid = ipfs_pin(image)
        image_hash = image_cid.split("/")[-1]

    # Create metadata JSON
    json_data = {
        "title": title,
        "description": tagline,
        "proposal": proposal,
        "eden_session_id": session_id,
        "attributes": [],
    }
    if image_hash:
        json_data["image"] = f"ipfs://{image_hash}"

    logger.info(f"Seed metadata: {json_data}")

    # Upload metadata to IPFS
    ipfs_hash = ipfs_pin(json_data)
    logger.info(f"Metadata uploaded to IPFS: {ipfs_hash}")

    # Prepare contract function call
    w3, owner, contract, abi = load_contract(
        address=contract_address,
        abi_path=CONTRACT_ABI_SEEDS,
        private_key=ABRAHAM_PRIVATE_KEY,
        network=Network.BASE_SEPOLIA,
    )

    # Pre-flight checks for better error messages
    _check_contract_access(w3, contract, owner.address)

    contract_function = contract.functions.submitSeed(f"ipfs://{ipfs_hash}")

    # Send transaction
    tx_hex, _ = safe_send(
        w3,
        contract_function,
        ABRAHAM_PRIVATE_KEY,
        op_name="SEED_SUBMIT",
        nonce=None,
        value=0,
        abi=abi,
        network=Network.BASE_SEPOLIA,
    )

    # Build explorer URL
    if not tx_hex.startswith("0x"):
        tx_hex = f"0x{tx_hex}"

    explorer_url = f"https://sepolia.basescan.org/tx/{tx_hex}"

    logger.info(f"Seed submitted successfully: {tx_hex}")
    logger.info(f"Explorer: {explorer_url}")

    return {
        "tx_hash": tx_hex,
        "ipfs_hash": ipfs_hash,
        "image_hash": image_hash,
        "explorer_url": explorer_url,
        "contract_address": contract_address,
    }


async def mint_seed_for_session(
    session: "Session",
    agent: "Agent",
    title: str,
    is_dm: bool = False,
) -> Optional[Seed]:
    """
    Create and mint a seed for a session with placeholder values.

    Args:
        session: The session to create a seed for
        agent: The agent that owns this seed
        title: The seed title (generated or "DM")
        is_dm: Whether this is a DM session
    """
    session_id = str(session.id)

    # Create seed document with pending status
    seed = Seed(
        session_id=ObjectId(session_id),
        agent_id=agent.id,
        title=title,
        tagline="Session seed new",
        proposal="Auto-generated session seed",
        image=None,
        status="pending",
        platform=session.platform,
        is_dm=is_dm,
    )
    seed.save()

    try:
        # Mint to blockchain
        result = mint_seed(
            title=title,
            tagline="Session seed",
            proposal="Auto-generated session seed",
            session_id=session_id,
            image=None,
        )

        # Update seed with blockchain metadata
        seed_metadata = SeedMetadata(
            contract_address=result["contract_address"],
            tx_hash=result["tx_hash"],
            ipfs_hash=result["ipfs_hash"],
            image_hash=result.get("image_hash"),
            explorer_url=result["explorer_url"],
            minted_at=datetime.now(),
        )

        seed.update(
            status="minted",
            seed_metadata=seed_metadata.model_dump(),
        )

        logger.info(f"Seed minted for session {session_id}: {result['explorer_url']}")
        return seed

    except Exception as e:
        logger.error(f"Failed to mint seed for session {session_id}: {e}")
        seed.update(status="failed")
        raise


async def maybe_create_session_seed(
    session: "Session",
    message_content: str,
) -> Optional[Seed]:
    """
    Hook function called after first user message.
    Creates a seed if the session's agent is opted-in.

    Args:
        session: The session that received a message
        message_content: The first user message content (for title generation)
    """
    from eve.agent import Agent
    from eve.agent.session.functions import async_title_session
    from eve.agent.session.models import ChatMessage

    # Check if seed already exists for this session
    existing = Seed.find_one({"session_id": session.id})
    if existing:
        logger.debug(f"Seed already exists for session {session.id}")
        return None

    # Count user messages to verify this is the first
    user_message_count = ChatMessage.get_collection().count_documents(
        {
            "session": session.id,
            "role": "user",
        }
    )
    if user_message_count != 1:
        logger.debug(f"Not first user message ({user_message_count}), skipping seed")
        return None

    # Check if any agent in session is seed-enabled
    enabled_agent = None
    for agent_id in session.agents:
        agent = Agent.from_mongo(agent_id)
        if agent and is_agent_seed_enabled(agent):
            enabled_agent = agent
            break

    if not enabled_agent:
        logger.debug(f"No seed-enabled agent in session {session.id}")
        return None

    # Determine title
    is_dm = is_dm_session(session)
    if is_dm:
        title = "DM"
    else:
        # Generate title and wait for completion
        title = await async_title_session(session, message_content)
        if not title:
            # Fallback if title generation fails
            session.reload()
            title = session.title or "Untitled"

    logger.info(f"Creating seed for session {session.id} with title: {title}")

    # Create and mint the seed
    return await mint_seed_for_session(
        session=session,
        agent=enabled_agent,
        title=title,
        is_dm=is_dm,
    )
