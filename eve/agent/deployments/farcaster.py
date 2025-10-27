import json
import os
import aiohttp
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from fastapi import Request
from fastapi.responses import JSONResponse

from eve.agent.session.models import (
    ChatMessageRequestInput,
    Session,
    SessionUpdateConfig,
    Deployment,
    DeploymentSecrets,
    DeploymentConfig,
)
from eve.api.api_requests import SessionCreationArgs
from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.deployments.neynar_client import NeynarClient
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.models import UpdateType
from eve.utils import prepare_result
from eve.user import User
import eve.mongo

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest

logger = logging.getLogger(__name__)


def uses_managed_signer(secrets: DeploymentSecrets) -> bool:
    """Check if deployment uses managed signer or mnemonic"""
    return bool(secrets.farcaster.signer_uuid)


async def get_fid(secrets: DeploymentSecrets) -> int:
    """Get FID based on auth method (managed signer or mnemonic)"""
    if uses_managed_signer(secrets):
        neynar_client = NeynarClient()
        user_info = await neynar_client.get_user_info_by_signer(
            secrets.farcaster.signer_uuid
        )
        return user_info.get("fid")
    else:
        from farcaster import Warpcast

        client = Warpcast(mnemonic=secrets.farcaster.mnemonic)
        user_info = client.get_me()
        return user_info.fid


async def post_cast(
    secrets: DeploymentSecrets,
    text: str = "",
    embeds: Optional[List[str]] = None,
    parent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Post a cast using either managed signer or mnemonic

    Returns dict with cast info including hash, url, thread_hash
    """
    logger.info(f"post_cast called - text: '{text[:100] if text else '(empty)'}...', embeds: {embeds}, parent: {parent}")

    if uses_managed_signer(secrets):
        # Use Neynar API for managed signer
        logger.info(f"Using managed signer with UUID: {secrets.farcaster.signer_uuid}")
        neynar_client = NeynarClient()
        result = await neynar_client.post_cast(
            signer_uuid=secrets.farcaster.signer_uuid,
            text=text,
            embeds=embeds,
            parent=parent,
        )

        logger.info(f"Neynar post_cast result: {result}")

        # Normalize Neynar response format
        cast_data = result.get("cast", {})
        cast_hash = cast_data.get("hash")
        author = cast_data.get("author", {})
        username = author.get("username")
        thread_hash = cast_data.get("thread_hash")

        cast_info = {
            "hash": cast_hash,
            "url": f"https://warpcast.com/{username}/{cast_hash}" if username and cast_hash else None,
            "thread_hash": thread_hash,
        }
        logger.info(f"Successfully posted cast via managed signer: {cast_info}")
        return cast_info
    else:
        # Use Warpcast client for mnemonic
        logger.info("Using mnemonic-based authentication")
        from farcaster import Warpcast

        client = Warpcast(mnemonic=secrets.farcaster.mnemonic)
        result = client.post_cast(text=text, embeds=embeds, parent=parent)

        # Convert to dict format for consistency
        user_info = client.get_me()
        cast_hash = result.cast.hash
        cast_info = {
            "hash": cast_hash,
            "url": f"https://warpcast.com/{user_info.username}/{cast_hash}",
            "thread_hash": result.cast.thread_hash,
        }
        logger.info(f"Successfully posted cast via mnemonic: {cast_info}")
        return cast_info


class FarcasterClient(PlatformClient):
    TOOLS = [
        "farcaster_cast",
        "farcaster_search",
        "farcaster_mentions",
    ]

    def _uses_managed_signer(self, secrets: DeploymentSecrets) -> bool:
        """Check if deployment uses managed signer or mnemonic"""
        return uses_managed_signer(secrets)

    async def _get_fid_from_managed_signer(self, signer_uuid: str) -> int:
        """Get FID from managed signer"""
        neynar_client = NeynarClient()
        user_info = await neynar_client.get_user_info_by_signer(signer_uuid)
        return user_info.get("fid")

    async def _get_fid_from_mnemonic(self, mnemonic: str) -> int:
        """Get FID from mnemonic using Warpcast client"""
        from farcaster import Warpcast

        client = Warpcast(mnemonic=mnemonic)
        user_info = client.get_me()
        return user_info.fid

    async def _get_fid(self, secrets: DeploymentSecrets) -> int:
        """Get FID based on auth method"""
        return await get_fid(secrets)

    async def _post_cast(
        self,
        text: str = "",
        embeds: Optional[List[str]] = None,
        parent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a cast using either managed signer or mnemonic"""
        if not self.deployment:
            raise ValueError("Deployment is required for _post_cast")

        return await post_cast(
            secrets=self.deployment.secrets,
            text=text,
            embeds=embeds,
            parent=parent,
        )

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Verify Farcaster credentials"""
        try:
            if self._uses_managed_signer(secrets):
                # Verify managed signer status
                neynar_client = NeynarClient()
                signer_status = await neynar_client.get_signer_status(
                    secrets.farcaster.signer_uuid
                )

                if signer_status.get("status") != "approved":
                    raise APIError(
                        f"Managed signer is not approved. Status: {signer_status.get('status')}",
                        status_code=400,
                    )
            else:
                # Verify mnemonic credentials
                from farcaster import Warpcast

                client = Warpcast(mnemonic=secrets.farcaster.mnemonic)

                # Test the credentials by getting user info
                client.get_me()
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Invalid Farcaster credentials: {str(e)}", status_code=400)

        # Generate webhook secret if not provided
        if not secrets.farcaster.neynar_webhook_secret:
            import secrets as python_secrets

            webhook_secret = python_secrets.token_urlsafe(32)
            secrets.farcaster.neynar_webhook_secret = webhook_secret

        try:
            # Add Farcaster tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Failed to add Farcaster tools: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """Add FID to existing webhook or skip if auto_reply disabled"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        # Only proceed if auto_reply is enabled
        if not (
            self.deployment.config
            and self.deployment.config.farcaster
            and self.deployment.config.farcaster.auto_reply
        ):
            return

        # Get webhook configuration from environment
        webhook_id = os.getenv("NEYNAR_WEBHOOK_ID")
        if not webhook_id:
            raise Exception("NEYNAR_WEBHOOK_ID not found in environment")

        # Get FID to add to webhook
        fid = await self._get_fid(self.deployment.secrets)

        await self._update_webhook_fids(webhook_id, add_fid=fid)

    async def update(
        self,
        old_config: Optional[DeploymentConfig] = None,
        new_config: Optional[DeploymentConfig] = None,
        old_secrets: Optional[DeploymentSecrets] = None,
        new_secrets: Optional[DeploymentSecrets] = None,
    ) -> None:
        """Handle deployment config/secrets updates"""
        if not self.deployment:
            raise ValueError("Deployment is required for update")

        # Check if auto_reply setting changed
        old_auto_reply = (
            old_config.farcaster.auto_reply
            if old_config and old_config.farcaster
            else False
        )
        new_auto_reply = (
            new_config.farcaster.auto_reply
            if new_config and new_config.farcaster
            else False
        )

        if old_auto_reply != new_auto_reply:
            webhook_id = os.getenv("NEYNAR_WEBHOOK_ID")
            if not webhook_id:
                raise Exception("NEYNAR_WEBHOOK_ID not found in environment")

            # Use new_secrets parameter (not self.deployment.secrets which is stale)
            secrets = new_secrets or self.deployment.secrets
            fid = await self._get_fid(secrets)

            if new_auto_reply and not old_auto_reply:
                # Enable auto-reply: add FID to webhook
                await self._update_webhook_fids(webhook_id, add_fid=fid)
            elif not new_auto_reply and old_auto_reply:
                # Disable auto-reply: remove FID from webhook
                await self._update_webhook_fids(webhook_id, remove_fid=fid)

    async def stop(self) -> None:
        """Stop Farcaster client by removing FID from webhook"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        # Get webhook configuration from environment
        webhook_id = os.getenv("NEYNAR_WEBHOOK_ID")
        if webhook_id:
            try:
                # Get FID to remove from webhook
                fid = await self._get_fid(self.deployment.secrets)

                await self._update_webhook_fids(webhook_id, remove_fid=fid)
            except Exception as e:
                logger.error(f"Error removing FID from Neynar webhook: {e}")

        try:
            # Remove Farcaster tools
            self.remove_tools()
        except Exception as e:
            logger.error(f"Failed to remove Farcaster tools: {e}")

    async def interact(self, request: Request) -> None:
        """Interact with the Farcaster client"""
        raise NotImplementedError(
            "Interact() with the Farcaster client is not supported"
        )

    async def handle_neynar_webhook(self, request: Request) -> None:
        """Interact with the Farcaster client"""
        import hmac
        import hashlib

        from eve.api.api_requests import PromptSessionRequest

        logger.info("=== Received Neynar webhook ===")

        # Verify Neynar webhook signature
        body = await request.body()
        signature = request.headers.get("X-Neynar-Signature")
        if not signature:
            logger.warning("Missing X-Neynar-Signature header")
            return JSONResponse(
                status_code=401, content={"error": "Missing signature header"}
            )

        # Find deployment by webhook secret - we'll store this in the deployment
        # For now, let's extract the webhook secret from headers or find another way
        webhook_data = await request.json()
        cast_data = webhook_data.get("data", {})

        logger.info(f"Webhook data received: {webhook_data}")

        if not cast_data or "hash" not in cast_data:
            logger.warning(f"Invalid cast data: {cast_data}")
            return JSONResponse(status_code=400, content={"error": "Invalid cast data"})

        # Use webhook secret from environment to verify signature
        webhook_secret = os.getenv("NEYNAR_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.error("NEYNAR_WEBHOOK_SECRET not configured in environment")
            return JSONResponse(
                status_code=401,
                content={"error": "Webhook secret not configured"},
            )

        # Verify signature
        computed_signature = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha512,
        ).hexdigest()

        if not hmac.compare_digest(computed_signature, signature):
            logger.warning(f"Invalid webhook signature. Expected: {computed_signature[:20]}..., Got: {signature[:20]}...")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid webhook signature"},
            )

        logger.info("Webhook signature verified successfully")

        # Find deployment by mentioned FID
        cast_author_fid = cast_data["author"]["fid"]
        cast_hash = cast_data["hash"]

        logger.info(f"Processing cast {cast_hash} from FID {cast_author_fid}")

        # Check if this is a self-mention (agent replying to itself) - ignore if so
        deployment = None
        farcaster_deployments = list(Deployment.find({"platform": "farcaster"}))
        logger.info(f"Found {len(farcaster_deployments)} Farcaster deployments")

        for d in farcaster_deployments:
            if d.config and d.config.farcaster and d.config.farcaster.auto_reply:
                try:
                    # Create temporary client instance to use helper methods
                    temp_client = FarcasterClient(deployment=d)
                    fid = await temp_client._get_fid(d.secrets)

                    logger.info(f"Checking deployment {d.id} with FID {fid}")

                    # Skip if this cast is from the agent itself (prevent loops)
                    if fid == cast_author_fid:
                        logger.info(f"Skipping deployment {d.id} - cast is from the agent itself")
                        return JSONResponse(status_code=200, content={"ok": True})

                    # Check if agent was mentioned or parent author
                    mentioned_fids = cast_data.get("mentioned_profiles", [])
                    mentioned_fid_list = [profile["fid"] for profile in mentioned_fids]

                    parent_cast = cast_data.get("parent_cast")
                    parent_author_fid = (
                        parent_cast.get("author", {}).get("fid")
                        if parent_cast
                        else None
                    )

                    logger.info(f"Mentioned FIDs: {mentioned_fid_list}, Parent author FID: {parent_author_fid}")

                    if fid in mentioned_fid_list or fid == parent_author_fid:
                        deployment = d
                        logger.info(f"Found matching deployment {d.id} for FID {fid}")
                        break

                except Exception as e:
                    logger.error(f"Error checking deployment {d.id}: {e}")
                    continue

        if not deployment:
            logger.warning("No matching deployment found for this cast")
            return JSONResponse(
                status_code=200,
                content={"ok": True, "message": "No matching deployment found"},
            )

        # Auto-reply check already done above in deployment selection

        # Create chat request similar to Telegram
        cast_hash = cast_data["hash"]
        author = cast_data["author"]
        author_username = author["username"]
        author_fid = author["fid"]

        logger.info(f"Processing cast from @{author_username} (FID: {author_fid})")

        # Get or create user
        user = User.from_farcaster(author_fid, author_username)
        logger.info(f"Got user {user.id} for FID {author_fid}")

        session_key = f"farcaster-{cast_hash}"

        # attempt to get session by session_key
        try:
            session = Session.load(session_key=session_key)
            logger.info(f"Found existing session {session.id} for cast {cast_hash}")

            # Check if the session is deleted or archived - if so, reactivate it
            needs_reactivation = False

            if hasattr(session, "deleted") and session.deleted:
                needs_reactivation = True
            elif hasattr(session, "status") and session.status == "archived":
                needs_reactivation = True

            if needs_reactivation:
                logger.info(f"Reactivating session {session.id}")
                session.deleted = False
                session.status = "active"
                session.save()
        except Exception as e:
            if isinstance(e, eve.mongo.MongoDocumentNotFound):
                logger.info(f"No existing session found for cast {cast_hash}, will create new one")
                session = None
            else:
                raise e

        cast_text = cast_data.get("text", "")
        logger.info(f"Cast text: '{cast_text}'")

        prompt_session_request = PromptSessionRequest(
            user_id=str(user.id),
            actor_agent_ids=[str(deployment.agent)],
            message=ChatMessageRequestInput(
                content=cast_text,
                sender_name=author_username,
            ),
            update_config=SessionUpdateConfig(
                deployment_id=str(deployment.id),
                update_endpoint=f"{get_api_url()}/v2/deployments/emission",
                farcaster_hash=cast_hash,
                farcaster_author_fid=author_fid,
            ),
        )

        # create session if it doesn't exist
        if session:
            prompt_session_request.session_id = str(session.id)
            logger.info(f"Using existing session {session.id}")
        else:
            prompt_session_request.creation_args = SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(deployment.agent)],
                title=f"Farcaster cast {cast_hash}",
                session_key=session_key,
                platform="farcaster",
            )
            logger.info(f"Will create new session with key {session_key}")

        # Make async HTTP POST to /chat
        eden_api_url = get_api_url()
        prompt_url = f"{eden_api_url}/sessions/prompt"
        logger.info(f"Sending prompt request to {prompt_url}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                prompt_url,
                json=prompt_session_request.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to process chat request: {error_text}")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"Failed to process chat request: {error_text}"
                        },
                    )
                else:
                    logger.info(f"Successfully sent prompt request, status: {response.status}")

        logger.info("=== Webhook processing complete ===")
        return JSONResponse(status_code=200, content={"ok": True})

    async def _update_webhook_fids(
        self, webhook_id: str, add_fid: int = None, remove_fid: int = None
    ) -> None:
        """Update webhook FID lists by adding or removing a FID"""
        neynar_api_key = os.getenv("NEYNAR_API_KEY")
        if not neynar_api_key:
            raise Exception("NEYNAR_API_KEY not found in environment")

        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": f"{neynar_api_key}",
                "Content-Type": "application/json",
            }

            # First get current webhook configuration
            async with session.get(
                f"https://api.neynar.com/v2/farcaster/webhook?webhook_id={webhook_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get webhook configuration: {error_text}"
                    )

                webhook_data = await response.json()

                current_subscription = webhook_data.get("webhook", {}).get(
                    "subscription", {}
                )

                # Handle both possible structures - filters.cast.created or cast.created
                filters = current_subscription.get("filters", {})
                cast_created = (
                    filters.get("cast.created", {})
                    if filters
                    else current_subscription.get("cast.created", {})
                )

                # Get current FID lists
                mentioned_fids = set(cast_created.get("mentioned_fids", []))
                parent_author_fids = set(cast_created.get("parent_author_fids", []))

                # Update FID lists
                if add_fid:
                    mentioned_fids.add(add_fid)
                    parent_author_fids.add(add_fid)

                if remove_fid:
                    mentioned_fids.discard(remove_fid)
                    parent_author_fids.discard(remove_fid)

                # Update webhook with new FID lists
                update_data = {
                    "webhook_id": webhook_id,
                    "url": os.getenv("NEYNAR_WEBHOOK_URL"),
                    "name": os.getenv("NEYNAR_WEBHOOK_NAME"),
                    "subscription": {
                        "cast.created": {
                            "mentioned_fids": list(mentioned_fids),
                            "parent_author_fids": list(parent_author_fids),
                        }
                    },
                }

                async with session.put(
                    "https://api.neynar.com/v2/farcaster/webhook",
                    headers=headers,
                    json=update_data,
                ) as update_response:
                    if update_response.status != 200:
                        error_text = await update_response.text()
                        raise Exception(f"Failed to update webhook: {error_text}")

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the platform client"""
        try:
            logger.info(f"=== Handling Farcaster emission: {emission.type} ===")

            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Extract context from update_config
            cast_hash = emission.update_config.farcaster_hash
            author_fid = emission.update_config.farcaster_author_fid

            if not cast_hash or not author_fid:
                logger.error(
                    "Missing farcaster_hash or farcaster_author_fid in update_config"
                )
                return

            logger.info(f"Replying to cast {cast_hash} from FID {author_fid}")

            update_type = emission.type

            if update_type == UpdateType.ASSISTANT_MESSAGE:
                content = emission.content
                logger.info(f"Processing ASSISTANT_MESSAGE with content length: {len(content) if content else 0}")
                if content:
                    try:
                        logger.info(f"Posting assistant message: '{content[:100]}...'")
                        result = await self._post_cast(
                            text=content,
                            parent={"hash": cast_hash, "fid": int(author_fid)},
                        )
                        logger.info(
                            f"Posted assistant message cast in reply to {cast_hash}. Result: {result}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to post cast: {str(e)}", exc_info=True)
                        raise
                else:
                    logger.warning("ASSISTANT_MESSAGE had no content to post")

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = emission.result
                logger.info(f"Processing TOOL_COMPLETE")
                if not result:
                    logger.debug("No tool result to post")
                    return

                # Process result to extract media URLs
                logger.info("Processing tool result: %s...", result[:200])
                processed_result = prepare_result(json.loads(result))
                logger.info("Processed result: %s", processed_result)

                if (
                    processed_result.get("result")
                    and len(processed_result["result"]) > 0
                    and "output" in processed_result["result"][0]
                ):
                    outputs = processed_result["result"][0]["output"]
                    logger.info(f"Found {len(outputs)} outputs in tool result")

                    # Extract URLs from outputs (up to 4 for Farcaster limit)
                    # Use creation page URLs for proper Open Graph video embedding
                    from eve.api.helpers import get_eden_creation_url

                    urls = []
                    for output in outputs[:4]:
                        if isinstance(output, dict):
                            # Prefer creation page URL for proper Open Graph video tags
                            if "creation" in output:
                                creation_url = get_eden_creation_url(str(output["creation"]))
                                urls.append(creation_url)
                            elif "url" in output:
                                urls.append(output["url"])

                    logger.info(f"Extracted {len(urls)} URLs from outputs: {urls}")

                    if urls:
                        try:
                            # Post cast with media embeds
                            logger.info(f"Posting cast with {len(urls)} media embeds")
                            result = await self._post_cast(
                                text="",  # Empty text, just media
                                embeds=urls,
                                parent={"hash": cast_hash, "fid": int(author_fid)},
                            )
                            logger.info(
                                f"Posted tool result cast with {len(urls)} embeds in reply to {cast_hash}. Result: {result}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to post cast with embeds: {str(e)}", exc_info=True)
                            raise
                    else:
                        logger.warning(
                            "No valid URLs found in tool result for Farcaster embeds"
                        )
                else:
                    logger.warning(
                        f"Unexpected tool result structure for Farcaster emission. Structure: {processed_result}"
                    )

            elif update_type == UpdateType.ERROR:
                error_msg = emission.error or "Unknown error occurred"
                logger.info(f"Processing ERROR emission: {error_msg}")
                try:
                    logger.info(f"Posting error message: {error_msg}")
                    result = await self._post_cast(
                        text=f"Error: {error_msg}",
                        parent={"hash": cast_hash, "fid": int(author_fid)},
                    )
                    logger.info(f"Posted error message cast in reply to {cast_hash}. Result: {result}")
                except Exception as e:
                    logger.error(f"Failed to post error cast: {str(e)}", exc_info=True)
                    # Don't re-raise for error posts to avoid infinite loops

            else:
                logger.info("Ignoring emission type: %s", update_type)

            logger.info("=== Emission handling complete ===")

        except Exception as e:
            logger.error(f"Error handling Farcaster emission: {str(e)}", exc_info=True)
            raise
