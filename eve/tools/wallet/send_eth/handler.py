from eve.tool import ToolContext
import os
from web3 import Web3


async def handler(context: ToolContext):
    # Initialize Web3 with Base Sepolia RPC URL
    w3 = Web3(Web3.HTTPProvider(os.getenv("BASE_SEPOLIA_RPC_URL")))
    
    # Get private key from environment
    private_key = os.getenv("WALLET_PRIVATE_KEY")
    if not private_key:
        raise Exception("Private key not found in environment variables")
    
    # Create account from private key
    account = w3.eth.account.from_key(private_key)
    
    # Prepare transaction
    receiver_address = context.args.get("receiver_address")
    amount_eth = float(context.args.get("amount_eth", 0))
    amount_wei = w3.to_wei(amount_eth, 'ether')
    
    tx = {
        'nonce': w3.eth.get_transaction_count(account.address),
        'to': receiver_address,
        'value': amount_wei,
        'gas': 21000,  # Standard ETH transfer
        'gasPrice': w3.eth.gas_price,
        'chainId': w3.eth.chain_id
    }
    
    # Sign and send transaction
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    
    return {
        "trasnaction_hash": tx_hash.hex()
    }
