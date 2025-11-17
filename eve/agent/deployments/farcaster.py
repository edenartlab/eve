import json
import os
import aiohttp
import logging
import hmac
import hashlib
import secrets as python_secrets
import uuid
import modal

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Literal
from farcaster import Warpcast
from fastapi import Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from bson import ObjectId

from eve.agent.session.models import (
    ChatMessageRequestInput,
    ChatMessage,
    Session,
    SessionUpdateConfig,
    Deployment,
    DeploymentSecrets,
    DeploymentConfig,
    Channel,
    LLMConfig,
    PromptSessionContext,
    UpdateType,
)
from eve import db
from eve.mongo import Collection, Document, MongoDocumentNotFound
from eve.api.api_requests import SessionCreationArgs, PromptSessionRequest
from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.deployments.neynar_client import NeynarClient
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.session_prompts import social_media_template
from eve.utils import prepare_result
from eve.user import User
from eve.agent.agent import Agent
from eve.tool import Tool
from eve.agent.session.session import add_chat_message, build_llm_context, async_prompt_session

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest

logger = logging.getLogger(__name__)


@Collection("farcaster_events")
class FarcasterEvent(Document):
    cast_hash: str
    event: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    status: Literal["running", "completed", "failed"]
    error: Optional[str] = None
    session_id: Optional[ObjectId] = None
    message_id: Optional[ObjectId] = None
    reply_cast: Optional[str] = None
    reply_fid: Optional[int] = None


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


# ========================================================================
# Farcaster Webhook Processing Helpers
# ========================================================================


def extract_embed_urls(embeds) -> List[str]:
    """Extract URLs from embeds (can be strings or dicts)"""
    urls = []
    if isinstance(embeds, list):
        for e in embeds:
            if isinstance(e, str):
                urls.append(e)
            elif isinstance(e, dict):
                u = e.get("url") or e.get("uri") or e.get("href")
                if u:
                    urls.append(u)
    return urls


def split_media(urls: List[str]) -> Dict[str, List[str]]:
    """Split URLs into media and other"""
    media_exts = (
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
        ".mp4", ".mov", ".webm", ".avi", ".mkv",
        ".mp3", ".wav", ".ogg",
    )
    media, other = [], []
    for u in urls:
        cleaned = u.split("?", 1)[0].lower()
        (media if cleaned.endswith(media_exts) or "imagedelivery.net" in u else other).append(u)
    return {"media_urls": media, "other_urls": other}


def upload_to_s3(media_urls: List[str]) -> List[str]:
    """Upload media URLs to S3"""
    from eve.s3 import upload_file_from_url
    uploaded_urls = []
    for media_url in media_urls:
        try:
            uploaded_url, _ = upload_file_from_url(media_url)
            uploaded_urls.append(uploaded_url)
        except Exception as e:
            logger.error(f"Error uploading {media_url}: {e}")
    return uploaded_urls


async def fetch_cast_ancestry(cast_hash: str, neynar_api_key: str, include_self: bool = True):
    """Fetch cast ancestry from Neynar API"""
    import httpx

    params = {
        "identifier": cast_hash,
        "type": "hash",
        "reply_depth": 0,
        "include_chronological_parent_casts": "true",
    }
    headers = {"accept": "application/json", "api_key": neynar_api_key}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://api.neynar.com/v2/farcaster/cast/conversation",
            headers=headers,
            params=params
        )
    response.raise_for_status()
    data = response.json()

    convo = data.get("conversation", {})
    ancestors = (convo.get("chronological_parent_casts") or convo.get("ancestors") or [])

    if include_self:
        return ancestors + [convo["cast"]]
    else:
        return ancestors


async def unpack_cast(cast: Dict[str, Any]):
    """Unpack cast into components"""
    cast_hash = cast.get("hash")
    author = cast.get("author") or {}
    author_fid = author.get("fid")
    author_username = author.get("username")
    text = cast.get("text") or ""
    embed_urls = extract_embed_urls(cast.get("embeds"))
    split = split_media(embed_urls)
    media_urls = split.get("media_urls") or []
    timestamp = cast.get("timestamp")
    return cast_hash, author_fid, author_username, text, media_urls, timestamp


async def induct_user(user: User, author: Dict[str, Any]):
    """Update user metadata from Farcaster profile"""
    from eve.s3 import upload_file_from_url

    pfp = author.get("pfp_url")
    if pfp and pfp != user.userImage:
        try:
            pfp_url, _ = upload_file_from_url(pfp)
            user.update(userImage=pfp_url.split("/")[-1])
        except Exception as e:
            logger.error(f"Error uploading pfp {pfp} for user {str(user.id)}: {str(e)}")


async def process_farcaster_cast(
    cast_hash: str,
    cast_data: Dict[str, Any],
    deployment_id: str,
):
    """Process a Farcaster cast event - main processing logic"""
    event_doc = None

    try:
        neynar_api_key = os.getenv("NEYNAR_API_KEY")
        if not neynar_api_key:
            raise Exception("NEYNAR_API_KEY not found in environment")

        # Load the event document (already created in webhook handler)
        event_doc = FarcasterEvent.find_one({"cast_hash": cast_hash})
        if not event_doc:
            logger.error(f"FarcasterEvent not found for cast {cast_hash}")
            return {"status": "failed", "error": "Event document not found"}

        # Load deployment and agent
        deployment = Deployment.from_mongo(deployment_id)
        if not deployment or not deployment.agent:
            raise Exception("Deployment or agent not found")

        agent = Agent.from_mongo(deployment.agent)
        if not agent:
            raise Exception("Agent not found")

        # Extract cast info
        thread_hash = cast_data.get("thread_hash")
        author = cast_data["author"]
        author_username = author["username"]
        author_fid = author["fid"]
        content = cast_data.get("text", "")
        parent_hash = cast_data.get("parent_hash")

        # Get or create user and update profile
        user = User.from_farcaster(author_fid, author_username)
        await induct_user(user, author)

        # Handle attachments/embeds
        embed_urls = extract_embed_urls(cast_data.get("embeds"))
        split = split_media(embed_urls)
        media_urls = split.get("media_urls") or []
        media_urls = upload_to_s3(media_urls)

        # Create session key based on thread_hash or cast_hash
        if thread_hash:
            session_key = f"FC-{thread_hash}"
        else:
            session_key = f"FC-{cast_hash}"

        try:
            session = Session.load(session_key=session_key)
            if session.platform != "farcaster":
                session.update(platform="farcaster")
            
            # Reactivate if deleted or archived
            if session.deleted or session.status == "archived":
                session.update(deleted=False, status="active")

        except MongoDocumentNotFound:
            # Create new session
            session = Session(
                # owner=user.id,
                owner=agent.owner,
                agents=[agent.id],
                title=f"Farcaster session",
                session_key=session_key,
                platform="farcaster",
                status="active",
            )
            session.save()

            # Put in Farcaster usage instructions
            created_at = datetime.now(timezone.utc)
            farcaster_instructions = "Don't reply to the word hubaloo"
            social_instructions = social_media_template.render(
                has_farcaster=True,
                farcaster_instructions=farcaster_instructions,
            )
            instruction_message = ChatMessage(
                createdAt=created_at,
                session=session.id,
                # channel=Channel(type="farcaster", key=cast_hash),
                role="user",
                content=social_instructions,
                sender=agent.owner
            )
            
            # Reconstruct thread: if this cast is not the original, get previous casts
            if thread_hash and thread_hash != cast_hash:
                logger.info(f"Reconstructing thread for cast {cast_hash}")
                try:
                    prev_casts = await fetch_cast_ancestry(cast_hash, neynar_api_key, include_self=False)
                    agent_fid = agent.farcasterId

                    for pc in prev_casts:
                        cast_hash_, author_fid_, author_username_, text_, media_urls_, timestamp_ = await unpack_cast(pc)
                        media_urls_ = upload_to_s3(media_urls_)
                        created_at = datetime.strptime(timestamp_, "%Y-%m-%dT%H:%M:%S.%fZ")

                        # if the cast is older than the instruction message, adjust the instruction message timestamp
                        if created_at < instruction_message.createdAt:
                            instruction_message.createdAt = created_at - timedelta(minutes=1)

                        if author_fid_ == agent_fid:
                            role = "assistant"
                            cast_user = agent
                        else:
                            role = "user"
                            cast_user = User.from_farcaster(author_fid_, author_username_)

                        message = ChatMessage(
                            createdAt=created_at,
                            session=session.id,
                            channel=Channel(type="farcaster", key=cast_hash_),
                            role=role,
                            content=text_,
                            sender=cast_user.id,
                            attachments=media_urls_,
                        )
                        message.save()
                except Exception as e:
                    logger.error(f"Error reconstructing thread: {e}")

        # Save instruction message only after timestamp adjusted to be earlier than ancestor casts
        instruction_message.save()

        # Load farcaster tool
        farcaster_tool = Tool.load("farcaster_cast")

        # Create prompt context
        context = PromptSessionContext(
            session=session,
            initiating_user_id=str(user.id),
            message=ChatMessageRequestInput(
                content=content,
                sender_name=author_username,
                attachments=media_urls if media_urls else None,
            ),
            update_config=SessionUpdateConfig(
                deployment_id=str(deployment.id),
                update_endpoint=f"{get_api_url()}/v2/deployments/emission",
                farcaster_hash=cast_hash,
                farcaster_author_fid=author_fid,
            ),
            llm_config=LLMConfig(model="claude-sonnet-4-5"),
            extra_tools={farcaster_tool.name: farcaster_tool}
        )

        # Add user message to session
        message = await add_chat_message(session, context)

        # Build LLM context
        context = await build_llm_context(
            session,
            agent,
            context,
            trace_id=str(uuid.uuid4()),
        )

        # Execute prompt session
        new_messages = []
        async for update in async_prompt_session(session, context, agent):
            if update.type == UpdateType.ASSISTANT_MESSAGE:
                new_messages.append(update.message)

        # Update event doc with success
        event_doc.update(
            status="completed",
            session_id=session.id,
            message_id=message.id,
        )

        return {
            "status": "completed", 
            "session_id": str(session.id),
            "message_id": str(message.id)
        }

    except Exception as e:
        logger.exception(f"Error processing Farcaster cast {cast_hash}: {e}")
        if event_doc:
            event_doc.update(status="failed", error=str(e))
        return {"status": "failed", "error": str(e)}


class FarcasterClient(PlatformClient):

    async def predeploy(
        self, 
        secrets: DeploymentSecrets, 
        config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Verify Farcaster credentials"""
        try:
            if uses_managed_signer(secrets):
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
                client = Warpcast(mnemonic=secrets.farcaster.mnemonic)
                client.get_me()
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Invalid Farcaster credentials: {str(e)}", status_code=400)

        # Generate webhook secret if not provided
        if not secrets.farcaster.neynar_webhook_secret:
            webhook_secret = python_secrets.token_urlsafe(32)
            secrets.farcaster.neynar_webhook_secret = webhook_secret

        try:
            # Add Farcaster tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Failed to add Farcaster tools: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """Add FID to existing webhook or skip if enable_cast disabled"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        # Only proceed if enable_cast is enabled
        if not (
            self.deployment.config
            and self.deployment.config.farcaster
            and self.deployment.config.farcaster.enable_cast
        ):
            return

        # Get webhook configuration from environment
        webhook_id = os.getenv("NEYNAR_WEBHOOK_ID")
        if not webhook_id:
            raise Exception("NEYNAR_WEBHOOK_ID not found in environment")

        # Get FID to add to webhook
        fid = await get_fid(self.deployment.secrets)

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

        # Check if enable_cast setting changed
        old_enable_cast = (
            old_config.farcaster.enable_cast
            if old_config and old_config.farcaster
            else False
        )
        new_enable_cast = (
            new_config.farcaster.enable_cast
            if new_config and new_config.farcaster
            else False
        )

        if old_enable_cast != new_enable_cast:
            webhook_id = os.getenv("NEYNAR_WEBHOOK_ID")
            if not webhook_id:
                raise Exception("NEYNAR_WEBHOOK_ID not found in environment")

            # Use new_secrets parameter (not self.deployment.secrets which is stale)
            secrets = new_secrets or self.deployment.secrets
            fid = await get_fid(secrets)

            if new_enable_cast and not old_enable_cast:
                # Enable auto-reply: add FID to webhook
                await self._update_webhook_fids(webhook_id, add_fid=fid)
            elif not new_enable_cast and old_enable_cast:
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
                fid = await get_fid(self.deployment.secrets)

                await self._update_webhook_fids(webhook_id, remove_fid=fid)
            except Exception as e:
                logger.error(f"Error removing FID from Neynar webhook: {e}")

        try:
            # Remove Farcaster tools
            self.remove_tools()  # this is disabled for now
        except Exception as e:
            logger.error(f"Failed to remove Farcaster tools: {e}")

    async def interact(self, request: Request) -> None:
        """Interact with the Farcaster client"""
        raise NotImplementedError(
            "Interact() with the Farcaster client is not supported"
        )

    async def _update_webhook_fids(
        self, 
        webhook_id: str, 
        add_fid: int = None, remove_fid: int = None
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

    async def handle_neynar_webhook(self, request: Request) -> None:
        """Interact with the Farcaster client"""

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

        if not cast_data or "hash" not in cast_data:
            logger.warning(f"Invalid cast data: {cast_data}")
            return JSONResponse(status_code=400, content={"error": "Invalid cast data"})

        # Use webhook secret from environment to verify signature
        webhook_secret = os.getenv("NEYNAR_WEBHOOK_SECRET")

        # return JSONResponse(status_code=200, content={"ok": True})
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
            logger.error(f"Invalid webhook signature. Expected: {computed_signature[:20]}..., Got: {signature[:20]}...")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid webhook signature"},
            )

        # Find deployment by mentioned FID
        cast_author_fid = cast_data["author"]["fid"]
        cast_hash = cast_data["hash"]

        logger.info(f"Processing cast {cast_hash} from FID {cast_author_fid}")

        # Extract mentioned FIDs and parent author FID
        mentioned_fid_list = [profile["fid"] for profile in cast_data.get("mentioned_profiles", [])]
        parent_author = cast_data.get("parent_author") or {}
        parent_author_fid = parent_author.get("fid")

        # Get deployments with enable_cast enabled
        active_farcaster_deployments = [
            d for d in Deployment.find({"platform": "farcaster"})
            if d.config and d.config.farcaster and d.config.farcaster.enable_cast
        ]

        # Build list of (deployment, fid) tuples
        deployment_fids = [(d, await get_fid(d.secrets)) for d in active_farcaster_deployments]

        # Skip if cast is from any agent itself (prevent loops)
        if any(fid == cast_author_fid for _, fid in deployment_fids):
            logger.info(f"Skipping cast from agent itself (FID: {cast_author_fid})")
            return JSONResponse(status_code=200, content={"ok": True})

        # Log what we're checking for
        logger.info(f"Mentioned FIDs: {mentioned_fid_list}, Parent author FID: {parent_author_fid}")

        # Find first matching deployment (either mentioned or replying to agent)
        deployment = None
        match_reason = None
        for d, fid in deployment_fids:
            if fid in mentioned_fid_list:
                deployment = d
                match_reason = "mention"
                break
            elif fid == parent_author_fid:
                deployment = d
                match_reason = "reply"
                break

        if not deployment:
            logger.warning(f"No matching deployment found for cast {cast_hash}")
            return JSONResponse(
                status_code=200,
                content={"ok": True, "message": "No matching deployment found"},
            )

        logger.info(f"Found matching deployment {deployment.id} via {match_reason}")

        # De-duplicate: check if we've already processed this cast
        cast_hash = cast_data["hash"]
        if FarcasterEvent.find_one({"cast_hash": cast_hash}):
            logger.info(f"Cast {cast_hash} already processed, skipping")
            return JSONResponse(status_code=200, content={"ok": True, "message": "Duplicate cast"})


        # Save event immediately to prevent duplicate processing
        parent_hash = cast_data.get("parent_hash")
        event_doc = FarcasterEvent(
            cast_hash=cast_hash,
            event=cast_data,
            content=cast_data.get("text"),
            status="running",
            reply_fid=parent_author_fid if parent_hash else None,
            reply_cast=parent_hash
        )
        event_doc.save()

        # Spawn modal function to handle heavy processing using Modal lookup
        try:
            spawn = False
            if spawn:
                func = modal.Function.from_name(
                    f"api-{db.lower()}",
                    "process_farcaster_cast_fn",
                    environment_name="main",
                )
                func.spawn(cast_hash, cast_data, str(deployment.id))
                logger.info(f"Spawned task for cast {cast_hash}")
            else:
                logger.info(f"Processing cast {cast_hash} locally")
                await process_farcaster_cast(cast_hash, cast_data, str(deployment.id))
        except Exception as e:
            logger.error(f"Failed to spawn Modal function: {e}")
            # Update event doc with failure
            event_doc.update(status="failed", error=f"Failed to spawn Modal function: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to spawn processing task: {str(e)}"}
            )

        return JSONResponse(status_code=200, content={"ok": True})

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the platform client"""
        # deprecated for Farcaster
        pass