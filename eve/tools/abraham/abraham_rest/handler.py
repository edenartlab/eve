import logging
import os

from eve.agent import Agent
from eve.tool import ToolContext
from eve.utils.chain_utils import (
    BlockchainError,
    Network,
    load_contract,
    safe_send,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CONTRACT_ADDRESS_COVENANT = os.getenv("CONTRACT_ADDRESS_COVENANT")
ABRAHAM_PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")

# ABI file is stored locally in the tool folder
CONTRACT_ABI_COVENANT = os.path.join(os.path.dirname(__file__), "abi_covenant.json")


def rest():
    try:
        if not ABRAHAM_PRIVATE_KEY:
            raise BlockchainError("ABRAHAM_PRIVATE_KEY not configured")
        if not CONTRACT_ADDRESS_COVENANT:
            raise BlockchainError("CONTRACT_ADDRESS_COVENANT not configured")

        w3, owner, contract, abi = load_contract(
            address=CONTRACT_ADDRESS_COVENANT,
            abi_path=CONTRACT_ABI_COVENANT,
            private_key=ABRAHAM_PRIVATE_KEY,
            # network=Network.ETH_SEPOLIA,
            network=Network.ETH_MAINNET,
        )

        # Prepare contract function call
        contract_function = contract.functions.takeRestDay()

        # Send transaction
        tx_hash, receipt = safe_send(
            w3,
            contract_function,
            ABRAHAM_PRIVATE_KEY,
            op_name="ABRAHAM_REST",
            nonce=None,
            value=0,
            abi=abi,
            # network=Network.ETH_SEPOLIA,
            network=Network.ETH_MAINNET,
        )

        # Build explorer URL for ETH Sepolia
        tx_hash_hex = tx_hash.hex()
        if not tx_hash_hex.startswith("0x"):
            tx_hash_hex = f"0x{tx_hash_hex}"
        # explorer_url = f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"
        explorer_url = f"https://etherscan.io/tx/{tx_hash_hex}"

        logger.info(f"✅ Rest committed successfully: {tx_hash_hex}")
        logger.info(f"Explorer: {explorer_url}")

        return {"tx_hash": tx_hash_hex, "explorer_url": explorer_url}

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_REST failed: {e}")
        raise


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    # Commit to blockchain
    try:
        result = rest()

        return {
            "output": [
                {"tx_hash": result["tx_hash"], "explorer_url": result["explorer_url"]}
            ]
        }
    except Exception as e:
        logger.error(f"Failed to rest: {e}")
        raise
