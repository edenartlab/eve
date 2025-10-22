import os
import aiohttp
import logging
from typing import Optional, Dict, Any, List
from farcaster import Warpcast

logger = logging.getLogger(__name__)


class NeynarClient:
    """Client for interacting with Neynar API for Farcaster managed signers"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEYNAR_API_KEY")
        if not self.api_key:
            raise ValueError("NEYNAR_API_KEY must be provided or set in environment")
        self.base_url = "https://api.neynar.com/v2"

    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for Neynar API requests"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def create_managed_signer(
        self, developer_mnemonic: str
    ) -> Dict[str, Any]:
        """
        Create a managed signer using the developer's mnemonic for sponsorship

        Args:
            developer_mnemonic: The developer's Farcaster mnemonic for signing

        Returns:
            Dict containing:
                - signer_uuid: The UUID of the created signer
                - public_key: The public key of the signer
                - status: The status of the signer
                - signer_approval_url: URL for user to approve the signer
        """
        async with aiohttp.ClientSession() as session:
            # Step 1: Create a signer
            async with session.post(
                f"{self.base_url}/farcaster/signer",
                headers=self._get_headers(),
                json={},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to create signer: {error_text}")

                signer_data = await response.json()
                signer_uuid = signer_data.get("signer_uuid")
                public_key = signer_data.get("public_key")

                if not signer_uuid or not public_key:
                    raise Exception(
                        f"Invalid response from Neynar: missing signer_uuid or public_key"
                    )

            # Step 2: Sign the public key with developer's mnemonic
            warpcast_client = Warpcast(mnemonic=developer_mnemonic)
            signature = warpcast_client.get_signer_signature(public_key)

            # Step 3: Register the signed key
            async with session.post(
                f"{self.base_url}/farcaster/signer/signed_key",
                headers=self._get_headers(),
                json={
                    "signer_uuid": signer_uuid,
                    "signature": signature.signature.hex(),
                    "app_fid": warpcast_client.fid,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to register signed key: {error_text}")

                signed_key_data = await response.json()

            # Step 4: Get the signer approval URL
            signer_approval_url = (
                f"https://warpcast.com/~/signer-approval?signer_uuid={signer_uuid}"
            )

            return {
                "signer_uuid": signer_uuid,
                "public_key": public_key,
                "status": signed_key_data.get("status", "pending_approval"),
                "signer_approval_url": signer_approval_url,
            }

    async def get_signer_status(self, signer_uuid: str) -> Dict[str, Any]:
        """
        Get the status of a managed signer

        Args:
            signer_uuid: The UUID of the signer

        Returns:
            Dict containing signer status information
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/farcaster/signer",
                headers=self._get_headers(),
                params={"signer_uuid": signer_uuid},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get signer status: {error_text}")

                return await response.json()

    async def get_user_info_by_signer(self, signer_uuid: str) -> Dict[str, Any]:
        """
        Get user info for a managed signer (after it's been approved)

        Args:
            signer_uuid: The UUID of the signer

        Returns:
            Dict containing user information including fid
        """
        signer_data = await self.get_signer_status(signer_uuid)

        if signer_data.get("status") != "approved":
            raise Exception(f"Signer is not approved, status: {signer_data.get('status')}")

        fid = signer_data.get("fid")
        if not fid:
            raise Exception("No FID found for approved signer")

        # Get user info from Neynar
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/farcaster/user/bulk",
                headers=self._get_headers(),
                params={"fids": str(fid)},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get user info: {error_text}")

                user_data = await response.json()
                users = user_data.get("users", [])
                if not users:
                    raise Exception(f"No user found for FID {fid}")

                return users[0]

    async def post_cast(
        self,
        signer_uuid: str,
        text: str,
        embeds: Optional[List[str]] = None,
        parent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Post a cast using a managed signer

        Args:
            signer_uuid: The UUID of the signer
            text: The text content of the cast
            embeds: List of URLs to embed (images, videos, etc.)
            parent: Parent cast info (for replies) with 'hash' and 'fid' keys

        Returns:
            Dict containing the posted cast information
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "signer_uuid": signer_uuid,
                "text": text,
            }

            if embeds:
                payload["embeds"] = [{"url": url} for url in embeds]

            if parent:
                # Neynar API expects parent as just the cast hash string
                # If parent is a dict (from Warpcast format), extract the hash and fid
                if isinstance(parent, dict):
                    payload["parent"] = parent["hash"]
                    # CRITICAL: Also send parent_author_fid for proper threading
                    if "fid" in parent:
                        payload["parent_author_fid"] = parent["fid"]
                else:
                    payload["parent"] = parent

            logger.info(f"Sending cast to Neynar with payload: {payload}")

            async with session.post(
                f"{self.base_url}/farcaster/cast",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                if response.status != 200 and response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Failed to post cast: {error_text}")

                result = await response.json()
                logger.info(f"Neynar cast response: {result}")

                # Fetch full cast details to verify threading
                cast_hash = result.get("cast", {}).get("hash")
                if cast_hash:
                    try:
                        async with session.get(
                            f"{self.base_url}/farcaster/cast",
                            headers=self._get_headers(),
                            params={"identifier": cast_hash, "type": "hash"},
                        ) as verify_response:
                            if verify_response.status == 200:
                                full_cast = await verify_response.json()
                                cast_data = full_cast.get("cast", {})
                                logger.info(f"Full cast verification - parent_hash: {cast_data.get('parent_hash')}, thread_hash: {cast_data.get('thread_hash')}")
                    except Exception as e:
                        logger.warning(f"Failed to verify full cast details: {e}")

                return result
