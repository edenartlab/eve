import json
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from eth_account import Account
from eth_utils import to_text
from web3 import Web3
from web3.exceptions import ContractLogicError, TimeExhausted

# Initialize logger
logger = logging.getLogger(__name__)

# Get RPC endpoints from environment variables
BASE_SEPOLIA_RPC = os.getenv("BASE_SEPOLIA_RPC")
ETH_SEPOLIA_RPC = os.getenv("ETH_SEPOLIA_RPC")
ETH_MAINNET_RPC = os.getenv("ETH_MAINNET_RPC")


class Network(Enum):
    BASE_SEPOLIA = "base_sepolia"
    ETH_SEPOLIA = "eth_sepolia"
    ETH_MAINNET = "eth_mainnet"


# ---------- Errors ----------


class BlockchainError(Exception):
    pass


# ---------- Helpers ----------


def _fmt_gwei(wei: int) -> str:
    return f"{wei / 1_000_000_000:.3f} gwei"


def _explorer_url(tx_hash_hex: str, network: Network = Network.BASE_SEPOLIA) -> str:
    if network == Network.BASE_SEPOLIA:
        return f"https://sepolia.basescan.org/tx/{tx_hash_hex}"
    elif network == Network.ETH_SEPOLIA:
        return f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"
    elif network == Network.ETH_MAINNET:
        return f"https://etherscan.io/tx/{tx_hash_hex}"
    else:
        raise ValueError(f"Unknown network: {network}")


def make_w3(network: Network = Network.BASE_SEPOLIA) -> Web3:
    if network == Network.BASE_SEPOLIA:
        rpc = BASE_SEPOLIA_RPC
    elif network == Network.ETH_SEPOLIA:
        rpc = ETH_SEPOLIA_RPC
    elif network == Network.ETH_MAINNET:
        rpc = ETH_MAINNET_RPC
    else:
        raise ValueError(f"Unknown network: {network}")
    if not rpc:
        raise BlockchainError(f"RPC endpoint not configured for {network.value}")
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise BlockchainError("Web3 connection failed")
    return w3


def resolve_nonce(w3: Web3, address: str, provided_nonce: Optional[int]) -> int:
    pending_nonce = w3.eth.get_transaction_count(address, "pending")
    if provided_nonce is None:
        return pending_nonce
    if provided_nonce > pending_nonce:
        logger.warning(
            f"Provided nonce {provided_nonce} > pending {pending_nonce}; tx will queue until gaps fill."
        )
    elif provided_nonce < pending_nonce:
        logger.warning(
            f"Provided nonce {provided_nonce} < pending {pending_nonce}; using {pending_nonce}."
        )
        return pending_nonce
    return provided_nonce


def suggest_fees(
    w3: Web3,
    urgency: str = "fast",
    min_priority_gwei: float = 1.25,
) -> Dict[str, int]:
    """
    EIP-1559 fee suggestion. Works on Base Sepolia where max_priority_fee may be 0.
    Returns either {'maxFeePerGas', 'maxPriorityFeePerGas} or {'gasPrice'} (legacy).
    """
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas")
    if base_fee is None:
        # Legacy (no EIP-1559). Small buffer + floor.
        gp = max(int(w3.eth.gas_price * 1.25), w3.to_wei(min_priority_gwei * 2, "gwei"))
        return {"gasPrice": gp}

    mult = {
        "slow": 1.2,
        "normal": 1.5,
        "fast": 2.0,
        "urgent": 3.0,
    }.get(urgency, 2.0)

    # Priority fee may be 0 on some RPCs; clamp to a sane minimum.
    try:
        prio = int(w3.eth.max_priority_fee)
    except Exception:
        prio = 0
    prio = max(prio, w3.to_wei(min_priority_gwei, "gwei"))

    # Compute max fee AFTER priority is finalized.
    max_fee = int(base_fee * mult) + prio

    # Final safety: never let maxFee < priority (violates EIP‑1559).
    if max_fee < prio:
        max_fee = prio

    return {"maxFeePerGas": max_fee, "maxPriorityFeePerGas": prio}


def decode_custom_error(error_data: str, abi: list) -> str:
    """
    Decode a custom Solidity error from hex data.
    Returns a human-readable error message with decoded parameters.
    """
    if isinstance(error_data, tuple):
        error_data = error_data[0]

    # Extract selector (first 4 bytes / 8 hex chars after 0x)
    selector = error_data[:10]
    params_data = error_data[10:]

    # Find matching error in ABI
    errors = [item for item in abi if item.get("type") == "error"]

    for error in errors:
        name = error["name"]
        inputs = error.get("inputs", [])

        # Build signature and calculate selector
        if inputs:
            param_types = ",".join(inp["type"] for inp in inputs)
            signature = f"{name}({param_types})"
        else:
            signature = f"{name}()"

        calc_selector = "0x" + Web3.keccak(text=signature).hex()[:8]

        if calc_selector == selector:
            # Decode parameters if any
            if inputs and params_data:
                try:
                    param_bytes = bytes.fromhex(params_data)
                    decoded_params = []

                    for i, inp in enumerate(inputs):
                        param_type = inp["type"]
                        param_name = inp.get("name", f"param{i}")

                        if param_type == "address":
                            offset = i * 32
                            addr_bytes = param_bytes[offset : offset + 32]
                            address = "0x" + addr_bytes[-20:].hex()
                            decoded_params.append(f"{param_name}={address}")
                        elif param_type.startswith("uint"):
                            offset = i * 32
                            value = int.from_bytes(
                                param_bytes[offset : offset + 32], "big"
                            )
                            decoded_params.append(f"{param_name}={value}")
                        else:
                            decoded_params.append(f"{param_name}=<{param_type}>")

                    return f"{name}({', '.join(decoded_params)})"
                except Exception:
                    return signature
            else:
                return f"{name}()"

    # Unknown error
    return f"Unknown error {selector}"


def _extract_revert_msg(err_dict: Dict[str, Any]) -> str:
    data = err_dict.get("data")
    if isinstance(data, str) and data.startswith("0x08c379a0"):  # Error(string)
        # Skip selector + offset boilerplate to get the string; or use eth_abi:
        try:
            # 4 bytes selector + 32 offset + 32 len = 68 bytes header
            msg_len = int(data[136 : 136 + 64], 16)
            raw = bytes.fromhex(data[200 : 200 + msg_len * 2])
            return to_text(raw)
        except Exception:
            pass
    return err_dict.get("message", "execution reverted")


def simulate_call(
    contract_function,
    from_address: str,
    value: int = 0,
    abi: Optional[list] = None,
) -> None:
    """
    Run a dry call to surface reverts pre-broadcast.
    If abi is provided, will decode custom errors for better error messages.
    """
    try:
        # Call with only essentials; don't pass gas/fees to avoid masking errors.
        contract_function.call({"from": from_address, "value": value})
    except ContractLogicError as e:
        # Extract error information from exception
        error_msg = str(e)
        error_data = None

        # ContractLogicError.args can be:
        # - ('execution reverted', 'no data') - no revert data
        # - ('execution reverted', '0x...') - with hex error data
        # - ('0x...',) - just hex data
        # - (('execution reverted', 'no data'),) - nested tuple
        if hasattr(e, "args") and e.args:
            first_arg = e.args[0]

            # Handle nested tuple case
            if isinstance(first_arg, tuple):
                if len(first_arg) >= 2:
                    potential_data = first_arg[1]
                    if isinstance(potential_data, str) and potential_data.startswith(
                        "0x"
                    ):
                        error_data = potential_data
                    elif potential_data == "no data":
                        error_msg = "execution reverted (no revert data - likely a require() without message, access control, or wrong contract address)"
            # Handle direct hex data
            elif isinstance(first_arg, str) and first_arg.startswith("0x"):
                error_data = first_arg
            # Handle ('execution reverted', '0x...') or ('execution reverted', 'no data')
            elif len(e.args) >= 2:
                second_arg = e.args[1]
                if isinstance(second_arg, str) and second_arg.startswith("0x"):
                    error_data = second_arg
                elif second_arg == "no data":
                    error_msg = "execution reverted (no revert data - likely a require() without message, access control, or wrong contract address)"

        # Try to decode custom error if we have hex data and ABI
        if error_data and abi:
            try:
                decoded = decode_custom_error(error_data, abi)
                error_msg = decoded
            except Exception:
                pass  # Fall back to original error message

        raise BlockchainError(f"Simulation reverted: {error_msg}") from e
    except ValueError as e:
        msg = (
            _extract_revert_msg(e.args[0])
            if e.args and isinstance(e.args[0], dict)
            else str(e)
        )
        raise BlockchainError(f"Simulation failed: {msg}") from e
    except Exception as e:
        raise BlockchainError(f"Simulation failed: {e}") from e


def estimate_gas(
    contract_function,
    from_address: str,
    nonce: int,
    value: int = 0,
    gas_limit_cap: int = 1_200_000,
    fee_params: Optional[Dict[str, int]] = None,
) -> int:
    """
    Try to get an accurate estimate. Add a small buffer and cap. If estimate fails,
    let caller decide a default fallback.
    """
    params = {"from": from_address, "nonce": nonce, "value": value}
    if fee_params and {"maxFeePerGas", "maxPriorityFeePerGas"} <= fee_params.keys():
        params |= {
            "maxFeePerGas": fee_params["maxFeePerGas"],
            "maxPriorityFeePerGas": fee_params["maxPriorityFeePerGas"],
        }
    est = contract_function.estimate_gas(params)
    return min(int(est * 1.20), gas_limit_cap)


def wait_for_confirmations(
    w3,
    tx_hash,
    confirmations=3,
    inclusion_timeout_s=120,
    conf_timeout_s=180,
    poll_interval=1.0,
):
    """
    Wait for inclusion, verify success, then wait N block confirmations.
    """
    tx_hex = tx_hash.hex()
    logger.info(f"Waiting for inclusion: {tx_hex} (timeout={inclusion_timeout_s}s)")
    try:
        receipt = w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=inclusion_timeout_s
        )
    except TimeExhausted as e:
        raise BlockchainError(
            f"Transaction not mined within {inclusion_timeout_s}s: {tx_hex}"
        ) from e
    if receipt.status != 1:
        raise BlockchainError(
            f"Transaction reverted in block {receipt.blockNumber}: {tx_hash.hex()}"
        )
    target = receipt.blockNumber + confirmations
    t0 = time.time()
    while w3.eth.block_number < target:
        if time.time() - t0 > conf_timeout_s:
            raise BlockchainError(
                f"Timed out waiting for {confirmations} confirmations: {tx_hash.hex()}"
            )
        time.sleep(poll_interval)
    return receipt


# ---------- Core: safe_send ----------


def safe_send(
    w3: Web3,
    contract_function,
    private_key: str,
    *,
    op_name: str = "TX",
    nonce: Optional[int] = None,
    value: int = 0,
    default_gas_limit: int = 250_000,
    gas_limit_cap: int = 5_000_000,
    confirmations: int = 3,
    timeout_s: int = 180,
    poll_interval: float = 1.0,
    urgency: str = "fast",
    speed_up_on_timeout: bool = True,
    speed_up_bump: float = 1.125,  # 12.5% bump when replacing
    abi: Optional[list] = None,  # Added to decode custom errors
    network: Network = Network.BASE_SEPOLIA,
) -> Tuple[bytes, Any]:
    """
    1) simulate, 2) estimate gas, 3) build+sign+send (EIP-1559 if available), 4) wait N confirmations.
    On inclusion timeout, optionally re-broadcast with higher fee using the same nonce.
    """

    # if True:   # temporarily babysit txs
    #     return "tx_simulated", "tx_simulated_receipt"

    owner_account = Account.from_key(private_key)
    addr = owner_account.address

    # ---- 1) simulate, revert if failure
    simulate_call(contract_function, addr, value=value, abi=abi)

    # ---- 2) nonce + gas + fees
    use_nonce = resolve_nonce(w3, addr, nonce)
    fee_params = suggest_fees(w3, urgency=urgency)

    try:
        gas_limit = estimate_gas(
            contract_function,
            addr,
            use_nonce,
            value=value,
            gas_limit_cap=gas_limit_cap,
            fee_params=fee_params,
        )
    except Exception as e:
        logger.warning(
            f"{op_name}: gas estimate failed: {e}; using default={default_gas_limit}"
        )
        gas_limit = default_gas_limit

    # Guard against RPC oddities: ensure maxFeePerGas >= maxPriorityFeePerGas
    if "maxFeePerGas" in fee_params and "maxPriorityFeePerGas" in fee_params:
        if fee_params["maxFeePerGas"] < fee_params["maxPriorityFeePerGas"]:
            latest_base = w3.eth.get_block("latest").get("baseFeePerGas", 0)
            # Lift maxFee to be at least prio + base fee (or prio + 0.1 gwei if base fee is tiny)
            fee_params["maxFeePerGas"] = fee_params["maxPriorityFeePerGas"] + max(
                latest_base, w3.to_wei(0.1, "gwei")
            )

    # ---- 3) build + sign + send
    common_fields = {
        "chainId": w3.eth.chain_id,
        "from": addr,
        "nonce": use_nonce,
        "gas": gas_limit,
        "value": value,
    }
    tx_fields = {**common_fields, **fee_params}
    if "maxFeePerGas" in fee_params and "maxPriorityFeePerGas" in fee_params:
        tx_fields["type"] = 2  # explicit EIP-1559

    tx = contract_function.build_transaction(tx_fields)
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)

    logger.info(
        f"→ {op_name} nonce={use_nonce} gas={gas_limit} "
        + (
            f"maxFee={_fmt_gwei(fee_params['maxFeePerGas'])} prio={_fmt_gwei(fee_params['maxPriorityFeePerGas'])}"
            if "maxFeePerGas" in fee_params
            else f"gasPrice={_fmt_gwei(fee_params['gasPrice'])}"
        )
    )

    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except ValueError as e:
        msg = (
            e.args[0].get("message")
            if e.args and isinstance(e.args[0], dict)
            else str(e)
        )
        if "replacement transaction underpriced" in msg.lower():
            raise BlockchainError(f"{op_name}: replacement underpriced") from e
        if "nonce too low" in msg.lower():
            raise BlockchainError(
                f"{op_name}: nonce too low (another tx likely mined)"
            ) from e
        if "insufficient funds" in msg.lower():
            raise BlockchainError(f"{op_name}: insufficient funds for gas") from e
        raise BlockchainError(f"{op_name}: broadcast error: {msg}") from e

    tx_hex = tx_hash.hex()
    logger.info(f"TX: {tx_hex}")
    logger.info(f"Explorer: {_explorer_url(tx_hex, network)}")

    # ---- 4) wait for confirmations, possibly speed up if stuck
    try:
        receipt = wait_for_confirmations(
            w3=w3,
            tx_hash=tx_hash,
            confirmations=confirmations,
            conf_timeout_s=timeout_s,
            inclusion_timeout_s=timeout_s,
            poll_interval=poll_interval,
        )

        eff = receipt.get("effectiveGasPrice")
        logger.info(
            f"✅ {op_name} confirmed: {tx_hex} | block={receipt.blockNumber} gasUsed={receipt.gasUsed}"
            + (f" effGasPrice={_fmt_gwei(eff)}" if eff is not None else "")
        )

        return tx_hex, receipt

    except BlockchainError as first_err:
        # Check if transaction reverted (already mined) vs timeout (still pending)
        if "reverted" in str(first_err).lower():
            # Transaction was mined but reverted - cannot replace with same nonce
            raise

        # Inclusion/confirmation timeout. Optionally replace with higher tip and try once or twice.
        if not speed_up_on_timeout:
            raise

        logger.warning(
            f"{op_name}: {first_err}. Attempting speed-up replacement (same nonce)."
        )

        # Fetch current head fees and bump
        attempts = 0
        last_err = first_err
        while attempts < 2:  # keep minimal (1–2 attempts)
            attempts += 1
            bump_fees = suggest_fees(w3, urgency="urgent")
            if "maxFeePerGas" in bump_fees:
                bump_fees["maxFeePerGas"] = int(
                    max(bump_fees["maxFeePerGas"], fee_params.get("maxFeePerGas", 0))
                    * speed_up_bump
                )
                bump_fees["maxPriorityFeePerGas"] = int(
                    max(
                        bump_fees["maxPriorityFeePerGas"],
                        fee_params.get("maxPriorityFeePerGas", 0),
                    )
                    * speed_up_bump
                )
                tx_fields = {**common_fields, **bump_fees, "type": 2}
            else:
                bump_fees["gasPrice"] = int(
                    max(bump_fees["gasPrice"], fee_params.get("gasPrice", 0))
                    * speed_up_bump
                )
                tx_fields = {**common_fields, **bump_fees}

            tx_replacement = contract_function.build_transaction(tx_fields)
            signed_replacement = w3.eth.account.sign_transaction(
                tx_replacement, private_key=private_key
            )

            try:
                tx_hash = w3.eth.send_raw_transaction(
                    signed_replacement.raw_transaction
                )
                tx_hex = tx_hash.hex()
                logger.info(
                    f"↻ Speed-up attempt {attempts}: {_fmt_gwei(bump_fees.get('maxFeePerGas', bump_fees.get('gasPrice', 0)))} → {tx_hex}  {_explorer_url(tx_hex, network)}"
                )
                receipt = wait_for_confirmations(
                    w3=w3,
                    tx_hash=tx_hash,
                    confirmations=confirmations,
                    conf_timeout_s=timeout_s,
                    inclusion_timeout_s=timeout_s,
                    poll_interval=poll_interval,
                )
                logger.info(
                    f"✅ {op_name} confirmed after speed-up: {tx_hex} | block={receipt.blockNumber}"
                )
                return tx_hex, receipt

            except ValueError as e:
                msg = (
                    e.args[0].get("message")
                    if e.args and isinstance(e.args[0], dict)
                    else str(e)
                )
                last_err = BlockchainError(f"Speed-up broadcast failed: {msg}")
                logger.warning(str(last_err))
                # Continue loop to try second bump
            except BlockchainError as e:
                last_err = e
                logger.warning(str(e))
                # Continue loop to try once more

        # If we got here, both attempts failed
        raise last_err


def load_contract(
    address: str,
    abi_path: str,
    private_key: str,
    network: Network = Network.BASE_SEPOLIA,
):
    """
    Loads a contract from an address and ABI.
    Returns: (w3, owner, contract, abi)
    """

    # Load ABI
    with open(abi_path, "r") as f:
        abi = json.load(f)

    w3 = make_w3(network)
    owner = Account.from_key(private_key)
    contract = w3.eth.contract(address=address, abi=abi)

    return w3, owner, contract, abi
