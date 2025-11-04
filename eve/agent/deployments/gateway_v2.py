import asyncio
import json
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import websockets
import aiohttp
from ably import AblyRealtime
from bson import ObjectId

from eve import db
from eve.agent.agent import Agent
from eve.user import User
from eve.agent.session.models import (
    ChatMessageRequestInput,
    Session,
    SessionUpdateConfig,
    Deployment,
    ClientType,
)
from eve.agent.session.file_config import SUPPORTED_NON_MEDIA_EXTENSIONS
from eve.agent.deployments.typing_manager import (
    DiscordTypingManager,
    TelegramTypingManager,
)
from eve.agent.deployments.gmail import (
    GmailClient,
    parse_inbound_email,
    unwrap_pubsub_message,
)
from eve.agent.deployments.utils import get_api_url
from eve.api.api_requests import SessionCreationArgs, PromptSessionRequest
from eve.api.errors import APIError
import eve.mongo
from fastapi import FastAPI, Request, HTTPException, Header
from contextlib import asynccontextmanager

# Override the imported db with uppercase version for Ably channel consistency
db = os.getenv("DB", "STAGE").upper()


def construct_agent_chat_url(agent_username: str) -> str:
    """
    Construct the Eden agent chat URL based on the environment and agent username.

    Args:
        agent_username: The username of the agent (not the ID)

    Returns:
        Properly formatted Eden agent chat URL
    """
    root_url = "app.eden.art" if db == "PROD" else "staging.app.eden.art"
    return f"https://{root_url}/chat/{agent_username}"


def _is_gmail_platform(platform: Any) -> bool:
    if isinstance(platform, ClientType):
        return platform == ClientType.GMAIL
    return str(platform) == ClientType.GMAIL.value


logger = logging.getLogger(__name__)
root_dir = Path(__file__).parent.parent.parent.parent


app_name = (
    f"discord-gateway-v2-{db}-{os.getenv('GATEWAY_ID')}"
    if os.getenv("GATEWAY_ID")
    else f"discord-gateway-v2-{db}"
)

# Create Modal app
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
        modal.Secret.from_name("eve-secrets-gateway", environment_name="main"),
    ],
)

# Set up image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libmagic1",
        "ffmpeg",
        "wget",
        "libnss3",
        "libnspr4",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libatspi2.0-0",
        "libxcomposite1",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .env({"DB": db})
    .env({"LOCAL_API_URL": os.getenv("LOCAL_API_URL") or ""})
    .env({"LOCAL_USER_ID": os.getenv("LOCAL_USER_ID") or ""})
    .add_local_python_source("eve", ignore=[])
)

WATCH_REFRESH_INTERVAL_HOURS = int(os.getenv("GMAIL_WATCH_REFRESH_HOURS", "6"))


class GatewayOpCode:
    DISPATCH = 0
    HEARTBEAT = 1
    IDENTIFY = 2
    RESUME = 6
    RECONNECT = 7
    INVALID_SESSION = 9
    HELLO = 10
    HEARTBEAT_ACK = 11


class GatewayEvent:
    READY = "READY"
    MESSAGE_CREATE = "MESSAGE_CREATE"


def _is_non_media_file_url(url: str) -> bool:
    """
    Check if a URL contains a non-media file extension pattern (e.g., .pdf?, .txt?).
    This is used to determine whether to use Discord's direct URL vs proxy_url.

    Args:
        url: The URL to check

    Returns:
        True if the URL contains a non-media file extension pattern, False otherwise
    """
    if not url:
        return False

    # Check for file extension patterns like .pdf?, .txt?, .csv?
    for ext in SUPPORTED_NON_MEDIA_EXTENSIONS:
        # Match pattern: extension followed by ? (query parameter)
        # Use re.escape to handle the dot in extension
        pattern = rf"{re.escape(ext)}\?"
        if re.search(pattern, url, re.IGNORECASE):
            return True

    return False


def _select_attachment_url(attachment: dict) -> str:
    """
    Select the appropriate URL for a Discord attachment.

    For non-media files (PDF, TXT, CSV, etc.), Discord's proxy_url may not work correctly,
    so we use the direct 'url' field. For images and videos, we prefer proxy_url.

    Args:
        attachment: Discord attachment object with 'url' and optionally 'proxy_url'

    Returns:
        The appropriate URL string, or None if no valid URL found
    """
    url = attachment.get("url")
    proxy_url = attachment.get("proxy_url")

    # If we have a URL and it's a non-media file, use the direct URL
    if url and _is_non_media_file_url(url):
        return url

    # Otherwise prefer proxy_url (for images/videos), fall back to url
    return proxy_url or url


class DiscordGatewayClient:
    GATEWAY_VERSION = 10
    GATEWAY_URL = f"wss://gateway.discord.gg/?v={GATEWAY_VERSION}&encoding=json"

    def __init__(self, deployment: Deployment):
        if deployment.platform != ClientType.DISCORD:
            raise ValueError("Deployment must be for Discord HTTP platform")

        self.deployment = deployment
        self.token = deployment.secrets.discord.token
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.heartbeat_interval: Optional[int] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect = True
        self._last_sequence = None
        self._session_id = None
        self.typing_manager = DiscordTypingManager(self)

        # Set up Ably for busy state updates
        self.ably_client = None
        self.busy_channel = None

    async def heartbeat_loop(self):
        try:
            while self._reconnect:
                if not self.ws:
                    logger.warning(
                        f"Heartbeat loop: WebSocket is None for {self.deployment.id}"
                    )
                    break
                try:
                    await self.ws.send(
                        json.dumps(
                            {"op": GatewayOpCode.HEARTBEAT, "d": self._last_sequence}
                        )
                    )
                    await asyncio.sleep(self.heartbeat_interval / 1000)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(
                        f"Heartbeat loop: Connection closed for {self.deployment.id}: {e}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Heartbeat loop error for {self.deployment.id}: {e}")
                    break
        except asyncio.CancelledError:
            logger.debug(f"Heartbeat loop cancelled for {self.deployment.id}")
            raise

    async def identify(self):
        logger.info(f"Identifying for deployment {self.deployment.id}")
        await self.ws.send(
            json.dumps(
                {
                    "op": GatewayOpCode.IDENTIFY,
                    "d": {
                        "token": self.token,
                        "intents": 1 << 9
                        | 1 << 12
                        | 1 << 15,  # GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT
                        "properties": {
                            "$os": "linux",
                            "$browser": "eve",
                            "$device": "eve",
                        },
                    },
                }
            )
        )

    async def resume(self):
        if not self._session_id or not self._last_sequence:
            return False

        try:
            logger.info(
                f"Resuming session for deployment {self.deployment.id} with session_id {self._session_id} and last_sequence {self._last_sequence}"
            )
            await self.ws.send(
                json.dumps(
                    {
                        "op": GatewayOpCode.RESUME,
                        "d": {
                            "token": self.token,
                            "session_id": self._session_id,
                            "seq": self._last_sequence,
                        },
                    }
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to resume session: {e}")
            return False

    def replace_mentions_with_usernames(
        self, message_content: str, mentions, prefix: str = "", suffix: str = ""
    ) -> str:
        """
        Replaces all mentions with their usernames.
        :param message_content: The message to replace mentions in.
        :param mentions: List of user mentions to replace
        :param prefix: Optional prefix to add before username
        :param suffix: Optional suffix to add after username
        :return: The message with all mentions replaced with their usernames.
        """
        import re

        for mention in mentions:
            message_content = re.sub(
                f"<@[!&]?{mention['id']}>",
                f"{prefix}{mention['username']}{suffix}",
                message_content,
            )
        return message_content.strip()

    def _parse_mentioned_agents(self, message_data: dict) -> list:
        """
        Parse mentioned users/roles and return agent IDs for any that are Discord bots (deployments).
        Uses database lookup to find matching deployments.
        :param message_data: Discord message data containing mentions and mention_roles
        :return: List of agent IDs that were mentioned
        """
        mentioned_agent_ids = []

        # Collect all Discord IDs from mentions and roles
        discord_ids = []

        # Check regular user mentions
        mentions = message_data.get("mentions", [])
        for mention in mentions:
            discord_id = mention.get("id")
            if discord_id:
                discord_ids.append(discord_id)

        # Check role mentions
        mention_roles = message_data.get("mention_roles", [])
        for role_id in mention_roles:
            if role_id:
                discord_ids.append(role_id)

        # If no Discord IDs to look up, return empty list
        if not discord_ids:
            return mentioned_agent_ids

        # Look up deployments that match these Discord application IDs
        try:
            deployments = list(
                Deployment.find(
                    {
                        "platform": ClientType.DISCORD.value,
                        "secrets.discord.application_id": {"$in": discord_ids},
                        "valid": {"$ne": False},
                    }
                )
            )

            for deployment in deployments:
                agent_id = str(deployment.agent)
                if agent_id not in mentioned_agent_ids:
                    mentioned_agent_ids.append(agent_id)
                    logger.info(
                        f"Found mentioned agent: {agent_id} (Discord app ID: {deployment.secrets.discord.application_id})"
                    )

        except Exception as e:
            logger.error(f"Error looking up mentioned agents: {e}")

        return mentioned_agent_ids

    async def _is_channel_allowlisted(
        self, channel_id: str, allowed_channels: list, trace_id: str
    ) -> bool:
        """
        Check if a channel is allowlisted. If the channel is not directly allowlisted,
        check if it's a thread and if its parent channel is allowlisted.
        """
        # First check if the channel is directly allowlisted
        if channel_id in allowed_channels:
            logger.info(
                f"[trace:{trace_id}] Channel {channel_id} is directly allowlisted"
            )
            return True

        # If not directly allowlisted, check if this might be a thread
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bot {self.token}",
                }
                url = f"https://discord.com/api/v10/channels/{channel_id}"

                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        channel_data = await response.json()
                        channel_type = channel_data.get("type")

                        # Discord thread types: 10 (GUILD_NEWS_THREAD), 11 (GUILD_PUBLIC_THREAD), 12 (GUILD_PRIVATE_THREAD)
                        if channel_type in [10, 11, 12]:
                            parent_id = channel_data.get("parent_id")
                            if parent_id and parent_id in allowed_channels:
                                logger.info(
                                    f"[trace:{trace_id}] Thread {channel_id} has allowlisted parent channel {parent_id}"
                                )
                                return True
                            else:
                                logger.info(
                                    f"[trace:{trace_id}] Thread {channel_id} has non-allowlisted parent channel {parent_id}"
                                )
                                return False
                        else:
                            logger.info(
                                f"[trace:{trace_id}] Channel {channel_id} is not a thread (type: {channel_type}) and not allowlisted"
                            )
                            return False
                    else:
                        logger.warning(
                            f"[trace:{trace_id}] Failed to fetch channel info for {channel_id}: {response.status}"
                        )
                        return False
        except Exception as e:
            logger.error(
                f"[trace:{trace_id}] Error checking if channel {channel_id} is allowlisted: {e}"
            )
            return False

    def _create_trace_id(self, message_data: dict) -> str:
        """Create a unique trace ID for this deployment-message combination"""
        return f"{self.deployment.id}-{message_data['id']}"

    def _should_skip_bot_message(self, message_data: dict, trace_id: str) -> bool:
        """Check if we should skip messages from the bot itself"""
        author_id = message_data.get("author", {}).get("id")
        if author_id == self.deployment.secrets.discord.application_id:
            logger.info(f"[{trace_id}] Skipping message from bot (author: {author_id})")
            return True
        return False

    async def _validate_deployment_config(self, trace_id: str) -> Optional[Deployment]:
        """Fetch and validate deployment configuration"""
        logger.debug(f"[{trace_id}] Validating deployment config")

        deployment = Deployment.from_mongo(str(self.deployment.id))
        if not deployment or not deployment.config:
            logger.warning(
                f"[{trace_id}] No config found for deployment {self.deployment.id}"
            )
            return None
        return deployment

    async def _check_channel_permissions(
        self, channel_id: str, deployment: Deployment, trace_id: str
    ) -> bool:
        """Check if the bot has permission to respond in this channel"""
        if not deployment.config.discord.channel_allowlist:
            logger.debug(
                f"[{trace_id}] No channel allowlist configured, allowing all channels"
            )
            return True

        allowed_channels = [
            item.id for item in deployment.config.discord.channel_allowlist
        ]
        logger.debug(
            f"[{trace_id}] Checking channel {channel_id} against allowlist: {allowed_channels}"
        )

        is_allowed = await self._is_channel_allowlisted(
            channel_id, allowed_channels, trace_id
        )
        if not is_allowed:
            logger.info(f"[{trace_id}] Channel {channel_id} not in allowlist")
        return is_allowed

    def _check_dm_enabled(self, deployment: Deployment) -> bool:
        """Check if DM responses are enabled for this deployment"""
        return (
            deployment.config
            and deployment.config.discord
            and deployment.config.discord.enable_discord_dm
        )

    async def _handle_direct_message(
        self, data: dict, trace_id: str, deployment: Deployment
    ) -> None:
        """Send a canned response for direct messages and exit early."""
        logger.info(f"[{trace_id}] Received DM, sending redirect message")

        agent_username: Optional[str] = None
        try:
            agent = Agent.from_mongo(deployment.agent)
            agent_username = agent.username
        except Exception as exc:
            logger.warning(
                f"[{trace_id}] Unable to load agent for DM redirect: {exc}",
                exc_info=True,
            )

        chat_url = (
            construct_agent_chat_url(agent_username)
            if agent_username
            else (
                f"https://{os.getenv('DB', 'STAGE') == 'PROD' and 'app.eden.art' or 'staging.app.eden.art'}/chat"
            )
        )

        channel_mentions = []
        if (
            deployment.config
            and deployment.config.discord
            and deployment.config.discord.channel_allowlist
        ):
            for item in deployment.config.discord.channel_allowlist:
                if item and item.id:
                    channel_mentions.append(f"<#{item.id}>")

        message_lines = [
            "Yoo, thanks for reaching out! For now, I can't continue chats in DMs yet...",
            f"Chat with me on Eden: {chat_url}",
        ]
        if channel_mentions:
            channels_text = ", ".join(channel_mentions)
            message_lines.append(f"Or ping me in: {channels_text}")
        else:
            message_lines.append(
                "Or say hi in the public Discord channels I'm active in."
            )

        payload = {
            "content": "\n".join(message_lines),
            "allowed_mentions": {"parse": []},
        }

        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://discord.com/api/v10/channels/{data['channel_id']}/messages"
                headers = {"Authorization": f"Bot {self.token}"}
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(
                            f"[{trace_id}] Failed to send DM redirect message: {error_text}"
                        )
                    else:
                        logger.info(
                            f"[{trace_id}] DM redirect message sent successfully"
                        )
        except Exception as exc:
            logger.error(
                f"[{trace_id}] Error sending DM redirect message: {exc}",
                exc_info=True,
            )

    def _extract_user_info(self, message_data: dict) -> Tuple[str, str, User]:
        """Extract user information from message data"""
        user_id = message_data.get("author", {}).get("id")
        username = message_data.get("author", {}).get("username", "User")
        user = User.from_discord(user_id, username)
        return user_id, username, user

    def _process_message_content(
        self, message_data: dict, is_dm: bool = False
    ) -> Tuple[str, list, bool, list]:
        """Process message content, mentions, and attachments"""
        logger.info("Processing message content", extra={"message_data": message_data})
        content = message_data["content"]
        mentioned_agent_ids = []
        logger.debug(f"message_data: {message_data}")

        # Handle mentions
        if "mentions" in message_data:
            content = self.replace_mentions_with_usernames(
                content, message_data["mentions"]
            )

        # Parse mentioned agents (now handles both mentions and mention_roles)
        mentioned_agent_ids = self._parse_mentioned_agents(message_data)

        # Handle references/replies
        if message_data.get("referenced_message"):
            ref_message = message_data["referenced_message"]
            ref_content = ref_message.get("content", "")
            content = f"(Replying to message: {ref_content[:100]} ...)\n\n{content}"

        # Get attachments - use smart URL selection based on file type
        attachments = []
        if "attachments" in message_data:
            for attachment in message_data["attachments"]:
                selected_url = _select_attachment_url(attachment)
                if selected_url:
                    attachments.append(selected_url)

        # Check if bot is mentioned (force reply) - check both mentions and mention_roles
        force_reply = False
        app_id = self.deployment.secrets.discord.application_id

        # For DMs, always force reply (no mention required)
        if is_dm:
            force_reply = True
        else:
            # Check user mentions
            if message_data.get("mentions") and any(
                mention.get("id") == app_id
                for mention in message_data.get("mentions", [])
            ):
                force_reply = True

            # Check role mentions
            if not force_reply and app_id in message_data.get("mention_roles", []):
                force_reply = True

            # Also check raw message content for mention patterns (handles improperly formatted tags)
            # This catches cases where users type mentions without autocomplete
            if not force_reply and app_id:
                mention_patterns = [
                    f"<@{app_id}>",  # User mention
                    f"<@!{app_id}>",  # User mention with nickname
                    f"<@&{app_id}>",  # Role mention
                ]
                original_content = message_data.get("content", "")
                if any(pattern in original_content for pattern in mention_patterns):
                    force_reply = True
                    logger.info(f"Detected mention in raw content for app_id {app_id}")

        content = content or "..."
        return content, attachments, force_reply, mentioned_agent_ids

    def _create_session_key(self, message_data: dict, user_id: str) -> str:
        """Create a session key based on Discord channel"""
        is_dm = message_data.get("guild_id") is None
        if is_dm:
            # Use consistent key format: discord-dm-{agent_id}-{user_id}
            return f"discord-dm-{self.deployment.agent}-{user_id}"
        else:
            guild_id = message_data.get("guild_id")
            channel_id = message_data.get("channel_id")
            return f"discord-{guild_id}-{channel_id}"

    async def _load_existing_session(
        self, session_key: str, trace_id: str
    ) -> Optional[Session]:
        """Try to load an existing session by key"""
        logger.debug(
            f"[{trace_id}] Looking for existing session with key: {session_key}"
        )

        try:
            session = Session.load(session_key=session_key)
            # Check if the session is deleted or archived - if so, reactivate it
            needs_reactivation = False

            if hasattr(session, "deleted") and session.deleted:
                needs_reactivation = True
            elif hasattr(session, "status") and session.status == "archived":
                needs_reactivation = True

            if needs_reactivation:
                session.deleted = False
                session.status = "active"
                session.save()

            return session
        except Exception as e:
            if isinstance(e, eve.mongo.MongoDocumentNotFound):
                logger.debug(
                    f"[{trace_id}] No existing session found for key: {session_key}"
                )
                return None
            else:
                logger.error(f"[{trace_id}] Error loading session: {e}")
                raise e

    async def _find_relevant_deployments(self, channel_id: str, trace_id: str) -> list:
        """Find all deployments that can respond to this channel"""
        logger.debug(
            f"[{trace_id}] Finding relevant deployments for channel {channel_id}"
        )

        all_deployments = list(
            Deployment.find(
                {
                    "platform": ClientType.DISCORD.value,
                    "valid": {"$ne": False},
                }
            )
        )

        relevant_deployments = []
        for deployment in all_deployments:
            if (
                deployment.config
                and deployment.config.discord
                and deployment.config.discord.channel_allowlist
            ):
                allowed_channels = [
                    item.id for item in deployment.config.discord.channel_allowlist
                ]

                # Check if channel is directly allowed or parent is allowed (for threads)
                channel_allowed = channel_id in allowed_channels

                if not channel_allowed:
                    # Check if it's a thread and parent is allowed
                    try:
                        async with aiohttp.ClientSession() as session:
                            headers = {
                                "Authorization": f"Bot {self.deployment.secrets.discord.token}"
                            }
                            url = f"https://discord.com/api/v10/channels/{channel_id}"
                            async with session.get(url, headers=headers) as response:
                                if response.status == 200:
                                    channel_data = await response.json()
                                    if channel_data.get("type") in [
                                        10,
                                        11,
                                        12,
                                    ]:  # Thread types
                                        parent_id = channel_data.get("parent_id")
                                        if parent_id and parent_id in allowed_channels:
                                            channel_allowed = True
                    except Exception:
                        pass  # Ignore errors in thread check

                if channel_allowed:
                    relevant_deployments.append(deployment)

        logger.debug(
            f"[{trace_id}] Found {len(relevant_deployments)} relevant deployments"
        )
        return relevant_deployments

    def _should_skip_for_agent_conflict(
        self, relevant_deployments: list, force_reply: bool, trace_id: str
    ) -> bool:
        """Check if this agent should skip due to multi-agent conflict resolution"""
        if len(relevant_deployments) <= 1:
            logger.debug(
                f"[{trace_id}] No agent conflicts (found {len(relevant_deployments)} deployments)"
            )
            return False

        if force_reply:
            logger.info(
                f"[{trace_id}] Bot mentioned - responding despite {len(relevant_deployments)} agents in channel"
            )
            return False

        # Sort by agent ID - lowest gets priority
        sorted_deployments = sorted(relevant_deployments, key=lambda d: str(d.agent))
        priority_agent = str(sorted_deployments[0].agent)
        current_agent = str(self.deployment.agent)

        if current_agent != priority_agent:
            logger.info(
                f"[{trace_id}] Skipping - agent {current_agent} not priority (priority: {priority_agent})"
            )
            return True

        logger.info(
            f"[{trace_id}] Proceeding - agent {current_agent} has priority among {len(relevant_deployments)} agents"
        )
        return False

    async def _handle_dm_routing(
        self, data: dict, trace_id: str, deployment: Deployment
    ) -> bool:
        """
        Handle DM-specific routing logic.
        Returns True if we should continue processing, False if we should exit.
        """
        if not self._check_dm_enabled(deployment):
            # Send canned response and exit
            logger.info(f"[{trace_id}] DM responses disabled, sending canned reply")
            await self._handle_direct_message(data, trace_id, deployment)
            return False

        # DM enabled, continue processing
        logger.info(f"[{trace_id}] DM responses enabled, processing as normal message")
        return True

    async def _should_process_channel_logic(
        self,
        is_dm: bool,
        channel_id: str,
        deployment: Deployment,
        force_reply: bool,
        trace_id: str,
    ) -> bool:
        """
        Determine if we should process channel-specific logic.
        Returns True if we should continue, False if we should skip.
        """
        # Skip all channel logic for DMs
        if is_dm:
            return True

        # Check channel permissions
        if not await self._check_channel_permissions(channel_id, deployment, trace_id):
            return False

        # Check for agent conflicts
        relevant_deployments = await self._find_relevant_deployments(
            channel_id, trace_id
        )
        if self._should_skip_for_agent_conflict(
            relevant_deployments, force_reply, trace_id
        ):
            return False

        return True

    def _create_session_request(
        self,
        user: User,
        username: str,
        content: str,
        attachments: list,
        channel_id: str,
        message_id: str,
        mentioned_agent_ids: list = None,
        is_dm: bool = False,
    ) -> PromptSessionRequest:
        """Create a PromptSessionRequest object"""
        # For DMs, always include this deployment's agent
        # For channels, if specific agents are mentioned, use those; otherwise empty list

        if is_dm:
            # For DMs, always ensure the deployment's agent responds
            actor_agent_ids = [str(self.deployment.agent)]
        elif mentioned_agent_ids:
            # For channels, use mentioned agents if any (deduplicate just in case)
            actor_agent_ids = list(dict.fromkeys(mentioned_agent_ids))
        else:
            # For channels with no mentions, use empty list
            actor_agent_ids = []

        return PromptSessionRequest(
            user_id=str(user.id),
            actor_agent_ids=actor_agent_ids,
            message=ChatMessageRequestInput(
                content=content,
                sender_name=username,
                attachments=attachments,
            ),
            update_config=SessionUpdateConfig(
                deployment_id=str(self.deployment.id),
                update_endpoint=f"{get_api_url()}/v2/deployments/emission",
                discord_channel_id=channel_id if not is_dm else None,
                discord_message_id=message_id,
                discord_user_id=user.discordId if is_dm else None,
            ),
        )

    async def _send_session_request(
        self, request: PromptSessionRequest, trace_id: str
    ) -> None:
        """Send the session request to the API"""
        api_url = get_api_url()
        logger.info(
            f"[{trace_id}] Sending session request to {api_url}/sessions/prompt"
        )
        logger.debug(f"[{trace_id}] Request payload: {request.model_dump()}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/sessions/prompt",
                json=request.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                    "X-Client-Platform": "discord",
                    "X-Client-Deployment-Id": str(self.deployment.id),
                    "X-Trace-Id": trace_id,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[{trace_id}] Session request failed: {error_text}")
                    raise Exception(f"Session request failed: {error_text}")
                else:
                    logger.info(f"[{trace_id}] Session request successful")

    async def handle_message(self, data: dict):
        """Main handler for Discord messages - coordinates the full message processing pipeline"""
        trace_id = self._create_trace_id(data)
        logger.info(
            f"[{trace_id}] Processing message from user {data.get('author', {}).get('username', 'Unknown')}"
        )

        # Skip all bot messages
        if data.get("author", {}).get("bot", False):
            logger.info(f"[{trace_id}] Skipping message from bot")
            return

        # Early exit checks
        if self._should_skip_bot_message(data, trace_id):
            return

        deployment = await self._validate_deployment_config(trace_id)
        if not deployment:
            return

        # Determine if this is a DM
        is_dm = not data.get("guild_id")

        # Handle DM routing
        if is_dm:
            if not await self._handle_dm_routing(data, trace_id, deployment):
                return

        # Extract message details
        channel_id = str(data["channel_id"])
        user_id, username, user = self._extract_user_info(data)
        content, attachments, force_reply, mentioned_agent_ids = (
            self._process_message_content(data, is_dm=is_dm)
        )
        logger.info(f"[{trace_id}] mentioned_agent_ids: {mentioned_agent_ids}")
        session_key = self._create_session_key(data, user_id)

        logger.debug(
            f"[{trace_id}] Session key: {session_key}, force_reply: {force_reply}, mentioned_agents: {mentioned_agent_ids}"
        )

        # Channel-specific logic (skipped for DMs)
        if not await self._should_process_channel_logic(
            is_dm, channel_id, deployment, force_reply, trace_id
        ):
            return

        # Load or create session
        session = await self._load_existing_session(session_key, trace_id)
        request = self._create_session_request(
            user,
            username,
            content,
            attachments,
            channel_id,
            str(data["id"]),
            mentioned_agent_ids,
            is_dm,
        )

        if session:
            request.session_id = str(session.id)
            logger.info(f"[{trace_id}] Using existing session: {session.id}")
        else:
            request.creation_args = SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(self.deployment.agent)],
                title=f"Discord {session_key}",
                session_key=session_key,
                platform="discord",
            )
            logger.info(f"[{trace_id}] Creating new session")

        # Send the request
        await self._send_session_request(request, trace_id)

    async def setup_ably(self):
        """Set up Ably for listening to busy state updates"""
        try:
            from ably import AblyRealtime

            self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
            channel_name = f"busy-state-discord-v2-{self.deployment.id}"
            self.busy_channel = self.ably_client.channels.get(channel_name)

            async def message_handler(message):
                trace_id = (
                    f"ably-{self.deployment.id}-{message.id}"  # Trace each Ably message
                )
                try:
                    data = message.data
                    if not isinstance(data, dict):
                        logger.warning(
                            f"[{trace_id}] Received non-dict message: {data}"
                        )
                        return

                    channel_id = data.get("channel_id")
                    request_id = data.get("request_id")
                    is_busy = data.get("is_busy")

                    # Log exactly what was received
                    logger.info(
                        f"[{trace_id}] Ably typing message: channel_id={channel_id}, "
                        f"request_id={request_id}, is_busy={is_busy}, data={data}"
                    )

                    if not channel_id or not request_id:
                        logger.warning(
                            f"[{trace_id}] Missing channel_id or request_id: {data}"
                        )
                        return

                    # Only proceed if is_busy is explicitly True or False
                    if is_busy is True:
                        channel_id_int = (
                            int(channel_id)
                            if isinstance(channel_id, str)
                            else channel_id
                        )
                        logger.info(
                            f"[{trace_id}] Starting typing - channel: {channel_id_int}, request: {request_id}"
                        )
                        await self.typing_manager.start_typing(
                            str(channel_id_int), request_id
                        )
                    elif is_busy is False:
                        channel_id_int = (
                            int(channel_id)
                            if isinstance(channel_id, str)
                            else channel_id
                        )
                        logger.info(
                            f"[{trace_id}] Stopping typing - channel: {channel_id_int}, request: {request_id}"
                        )
                        await self.typing_manager.stop_typing(
                            str(channel_id_int),
                            reason="ably_signal",
                            request_id=request_id,
                        )
                    else:
                        logger.warning(
                            f"[{trace_id}] Invalid is_busy value ({is_busy}): {data}"
                        )

                except Exception as e:
                    logger.error(
                        f"[{trace_id}] Error handling Ably message: {e}", exc_info=True
                    )

            await self.busy_channel.subscribe(message_handler)
            logger.info(f"Subscribed to Ably busy state updates: {channel_name}")

        except Exception as e:
            logger.error(
                f"Failed to setup Ably for {self.deployment.id}: {e}", exc_info=True
            )

    async def connect(self):
        # Set up Ably first
        await self.setup_ably()
        # Start the typing manager's cleanup loop
        self.typing_manager.start_cleanup_loop()

        while self._reconnect:
            try:
                logger.info(
                    f"Connecting to gateway for deployment {self.deployment.id}"
                )
                async with websockets.connect(
                    self.GATEWAY_URL, ping_interval=None
                ) as ws:  # Disable automatic pings
                    self.ws = ws

                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data["op"] != GatewayOpCode.HELLO:
                        raise Exception(f"Expected HELLO, got {data}")

                    self.heartbeat_interval = data["d"]["heartbeat_interval"]
                    self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())

                    if not await self.resume():
                        await self.identify()

                    async for message in ws:
                        data = json.loads(message)

                        if data.get("s"):
                            self._last_sequence = data["s"]

                        # Check for authentication failures
                        if data.get("op") == GatewayOpCode.INVALID_SESSION:
                            # Check the close code if available
                            d_data = data.get("d", {})
                            close_code = (
                                d_data.get("code", 0) if isinstance(d_data, dict) else 0
                            )
                            if close_code == 4004 or "Authentication failed" in str(
                                data
                            ):
                                logger.error(
                                    f"Authentication failed for deployment {self.deployment.id}: {data}"
                                )
                                # Mark the deployment as invalid in the database
                                self._mark_deployment_invalid()
                                # Stop reconnection attempts
                                self._reconnect = False
                                break

                        if data["op"] == GatewayOpCode.DISPATCH:
                            if data["t"] == GatewayEvent.MESSAGE_CREATE:
                                # For message creation, use deployment ID and message ID for trace ID
                                message_id = data["d"].get("id", "unknown")
                                msg_trace_id = f"{self.deployment.id}-{message_id}"
                                logger.info(
                                    f"[trace:{msg_trace_id}] Dispatch event for deployment {self.deployment.id}"
                                )
                                logger.info(
                                    f"[trace:{msg_trace_id}] Handling message create event for deployment {self.deployment.id}"
                                )
                                await self.handle_message(data["d"])
                            elif data["t"] == GatewayEvent.READY:
                                logger.info(
                                    f"Ready event for deployment {self.deployment.id}"
                                )
                                self._session_id = data["d"]["session_id"]

                                # Set application_id if not already set
                                if not self.deployment.secrets.discord.application_id:
                                    self.deployment.secrets.discord.application_id = (
                                        data["d"]["application"]["id"]
                                    )
                                    self.deployment.save()
                                    logger.info(
                                        f"Set application_id to {self.deployment.secrets.discord.application_id} for deployment {self.deployment.id}"
                                    )

                                logger.info(
                                    f"Gateway connected for deployment {self.deployment.id}"
                                )

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
            ) as e:
                logger.error(
                    f"Gateway connection closed for {self.deployment.id}: Code={e.code}, Reason={e.reason}"
                )
                # Perform cleanup before attempting reconnect or stopping
                await self.cleanup_resources()
                # Check for authentication failure (code 4004)
                if e.code == 4004 or "Authentication failed" in str(e):
                    logger.error(
                        f"Authentication failed for deployment {self.deployment.id}: {e}"
                    )
                    # Mark the deployment as invalid in the database
                    self._mark_deployment_invalid()
                    # Stop reconnection attempts
                    self._reconnect = False
                    break
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(
                    f"Gateway connection error for {self.deployment.id}: {e}",
                    exc_info=True,
                )
                # Perform cleanup before attempting reconnect or stopping
                await self.cleanup_resources()
                # Check if this is an authentication failure
                if "4004" in str(e) or "Authentication failed" in str(e):
                    logger.error(
                        f"Authentication failed for deployment {self.deployment.id}, marking as invalid"
                    )
                    # Mark the deployment as invalid in the database
                    self._mark_deployment_invalid()
                    # Stop reconnection attempts
                    self._reconnect = False
                    break
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                await asyncio.sleep(5)
            finally:
                # Ensure cleanup if loop exits
                if not self._reconnect:
                    logger.info(
                        f"Reconnect disabled for {self.deployment.id}, ensuring final cleanup."
                    )
                    await self.cleanup_resources()
                # Ensure websocket is closed if loop exits while connected
                if self.ws:
                    try:
                        # Check if websocket is still open using state property
                        if hasattr(self.ws, "state") and self.ws.state.name in [
                            "OPEN",
                            "CLOSING",
                        ]:
                            await self.ws.close()
                        elif (
                            hasattr(self.ws, "close_code")
                            and self.ws.close_code is None
                        ):
                            # Alternative check - if no close code, might still be open
                            await self.ws.close()
                    except Exception as e:
                        logger.debug(f"WebSocket already closed or error closing: {e}")
                    finally:
                        self.ws = None

    async def cleanup_resources(self):
        """Clean up heartbeat task and typing manager."""
        logger.info(f"Cleaning up resources for deployment {self.deployment.id}")
        # Stop heartbeat task
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self.heartbeat_task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout waiting for heartbeat task cancellation for {self.deployment.id}"
                )
            except Exception as e:
                logger.error(f"Error awaiting heartbeat task cancellation: {e}")
        self.heartbeat_task = None
        # Cleanup typing manager
        try:
            await self.typing_manager.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up typing manager: {e}")
        # Close Ably client (if managed per-client)
        if self.ably_client:
            logger.info(f"Closing Ably client for {self.deployment.id}")
            try:
                await self.ably_client.close()
            except Exception as e:
                logger.error(f"Error closing Ably client for {self.deployment.id}: {e}")
            self.ably_client = None

    def _mark_deployment_invalid(self):
        """Mark the deployment as invalid in the database"""
        try:
            # Fetch fresh deployment data to avoid overwriting other changes
            deployment = Deployment.from_mongo(str(self.deployment.id))
            if deployment:
                deployment.valid = False
                deployment.save()
                logger.info(
                    f"Marked deployment {self.deployment.id} as invalid due to authentication failure"
                )
        except Exception as e:
            logger.error(
                f"Failed to mark deployment {self.deployment.id} as invalid: {e}"
            )

    def stop(self):
        """Initiates the stop sequence for the gateway client."""
        logger.info(f"Initiating stop for gateway client {self.deployment.id}")
        self._reconnect = False  # Prevent automatic reconnections

        # Close websocket if open - this will trigger cleanup in connect() finally block
        if self.ws:
            try:
                # Check if websocket is still open
                ws_is_open = False
                if hasattr(self.ws, "state"):
                    # websockets library uses state property
                    ws_is_open = self.ws.state.name in ["OPEN", "CLOSING"]
                elif hasattr(self.ws, "close_code"):
                    # Check if close_code is None (still open)
                    ws_is_open = self.ws.close_code is None

                if ws_is_open:
                    logger.info(
                        f"Closing WebSocket connection for {self.deployment.id}"
                    )
                    # Schedule the close, don't block here
                    asyncio.create_task(
                        self.ws.close(code=1000, reason="Client stopping")
                    )
                else:
                    logger.info(f"WebSocket already closed for {self.deployment.id}")
                    # Trigger cleanup since WS is already closed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.cleanup_resources())
                    except Exception as e:
                        logger.error(f"Error scheduling cleanup: {e}")
            except Exception as e:
                logger.error(
                    f"Error checking WebSocket state for {self.deployment.id}: {e}"
                )
                # Try to trigger cleanup anyway
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.cleanup_resources())
                except Exception as cleanup_error:
                    logger.error(f"Error scheduling cleanup: {cleanup_error}")
        else:
            # No WebSocket connection, just trigger cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logger.info(
                        f"No WS for {self.deployment.id}, scheduling resource cleanup."
                    )
                    asyncio.create_task(self.cleanup_resources())
            except Exception as e:
                logger.error(
                    f"Error triggering cleanup in stop() for {self.deployment.id}: {e}"
                )

        # Cancel heartbeat task directly as well, in case connect() loop isn't entered/exited cleanly
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()


class GatewayManager:
    def __init__(self):
        self.clients: Dict[str, DiscordGatewayClient] = {}
        self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        self.channel = self.ably_client.channels.get(f"discord-gateway-v2-{db}")

        # Add Telegram typing manager - use improved version
        self.telegram_typing_manager = TelegramTypingManager()

        # Set up Ably for Telegram busy state updates
        self.telegram_busy_channel = None

    async def reload_client(self, deployment_id: str):
        """Reload a gateway client with fresh deployment data"""
        reload_trace_id = (
            f"{deployment_id}-reload-{int(asyncio.get_event_loop().time())}"
        )
        logger.info(
            f"[trace:{reload_trace_id}] Reloading gateway client for deployment {deployment_id}"
        )

        # First stop and remove the existing client if it exists
        if deployment_id in self.clients:
            client = self.clients.pop(deployment_id)
            client.stop()
            logger.info(
                f"[trace:{reload_trace_id}] Stopped existing client for deployment {deployment_id}"
            )

            # Add a small delay to ensure cleanup
            await asyncio.sleep(1)

        # Get fresh deployment data from database
        deployment = Deployment.from_mongo(deployment_id)
        if deployment:
            # Check if the deployment is marked as invalid
            if deployment.valid is False:
                logger.info(
                    f"[trace:{reload_trace_id}] Skipping invalid deployment {deployment_id}"
                )
                return

            # Create a completely new client with the fresh data
            client = DiscordGatewayClient(deployment)
            self.clients[deployment_id] = client

            # Start the new client
            asyncio.create_task(client.connect())
            logger.info(
                f"[trace:{reload_trace_id}] Successfully reloaded gateway client for deployment {deployment_id} with fresh data"
            )

            # Log the updated channel allowlist for debugging
            if (
                deployment.config
                and deployment.config.discord
                and deployment.config.discord.channel_allowlist
            ):
                allowed_channels = [
                    item.id for item in deployment.config.discord.channel_allowlist
                ]
                logger.info(
                    f"[trace:{reload_trace_id}] Updated allowlist for deployment {deployment_id}: {allowed_channels}"
                )
        else:
            logger.error(
                f"[trace:{reload_trace_id}] Failed to reload - deployment not found: {deployment_id}"
            )

    async def setup_ably(self):
        """Set up Ably subscription for gateway commands"""

        async def message_handler(message):
            try:
                data = message.data
                if not isinstance(data, dict):
                    logger.warning(f"Received invalid message format: {data}")
                    return

                # Continue with existing command handling
                command = data.get("command")
                deployment_id = data.get("deployment_id")
                # Generate trace ID for command handling with deployment ID and timestamp
                cmd_trace_id = (
                    f"{deployment_id}-cmd-{int(asyncio.get_event_loop().time())}"
                )

                if not command or not deployment_id:
                    logger.warning(
                        f"[trace:{cmd_trace_id}] Missing command or deployment_id: {data}"
                    )
                    return

                logger.info(
                    f"[trace:{cmd_trace_id}] Received command: {command} for deployment: {deployment_id}"
                )

                if command == "start":
                    # Start a new gateway client
                    deployment = Deployment.from_mongo(deployment_id)
                    if deployment:
                        # Skip if the deployment is marked as invalid
                        if deployment.valid is False:
                            logger.info(
                                f"[trace:{cmd_trace_id}] Skipping invalid deployment: {deployment_id}"
                            )
                        else:
                            await self.start_client(deployment)
                            logger.info(
                                f"[trace:{cmd_trace_id}] Started client for deployment: {deployment_id}"
                            )
                    else:
                        logger.error(
                            f"[trace:{cmd_trace_id}] Deployment not found: {deployment_id}"
                        )

                elif command == "stop":
                    # Stop an existing gateway client
                    await self.stop_client(deployment_id)
                    logger.info(
                        f"[trace:{cmd_trace_id}] Stopped client for deployment: {deployment_id}"
                    )

                # Add Telegram-specific commands
                elif command == "register_telegram":
                    # Register a new Telegram deployment
                    token = data.get("token")
                    if token:
                        self.telegram_typing_manager.register_deployment(
                            deployment_id, token
                        )
                        logger.info(
                            f"[trace:{cmd_trace_id}] Registered Telegram deployment: {deployment_id}"
                        )
                    else:
                        logger.error(
                            f"[trace:{cmd_trace_id}] Missing token for Telegram registration: {data}"
                        )

                elif command == "unregister_telegram":
                    # Unregister a Telegram deployment
                    self.telegram_typing_manager.unregister_deployment(deployment_id)
                    logger.info(
                        f"[trace:{cmd_trace_id}] Unregistered Telegram deployment: {deployment_id}"
                    )

            except Exception as e:
                logger.error(f"Error handling Ably message: {e}")
                logger.exception(e)

        # Subscribe to the channel
        await self.channel.subscribe(message_handler)
        logger.info("Subscribed to Ably channel for gateway commands")

        # Set up Ably for Telegram busy state updates
        try:
            # Subscribe to Telegram busy state updates
            telegram_channel = self.ably_client.channels.get(
                f"busy-state-telegram-{db}"
            )

            async def telegram_message_handler(message):
                try:
                    data = message.data
                    if not isinstance(data, dict):
                        return

                    deployment_id = data.get("deployment_id")
                    chat_id = data.get("chat_id")
                    thread_id = data.get("thread_id")
                    request_id = data.get("request_id")
                    is_busy = data.get("is_busy")

                    logger.info(
                        f"Telegram typing update: deployment={deployment_id}, "
                        f"chat={chat_id}, request={request_id}, busy={is_busy}"
                    )

                    if deployment_id and chat_id and request_id:
                        if is_busy is True:
                            await self.telegram_typing_manager.start_typing_with_deployment(
                                deployment_id, str(chat_id), request_id, thread_id
                            )
                        elif is_busy is False:
                            channel_key = (
                                f"{deployment_id}:{chat_id}:{thread_id or 'main'}"
                            )
                            await self.telegram_typing_manager.stop_typing(
                                channel_key, reason="ably_signal", request_id=request_id
                            )
                except Exception as e:
                    logger.error(f"Error handling Telegram busy state update: {e}")

            await telegram_channel.subscribe(telegram_message_handler)
            logger.info("Subscribed to Telegram busy state updates")

        except Exception as e:
            logger.error(f"Failed to setup Telegram Ably subscription: {e}")

    async def load_deployments(self):
        """Load all Discord HTTP deployments from database"""
        from bson import ObjectId as BsonObjectId

        # Filter deployments based on LOCAL_API_URL environment variable
        local_api_url = os.getenv("LOCAL_API_URL")
        local_user_id = os.getenv("LOCAL_USER_ID")

        if local_api_url and local_api_url != "":
            # If LOCAL_API_URL is set, only load local deployments
            deployment_filter = {"platform": ClientType.DISCORD.value, "local": True}
            logger.info("LOCAL_API_URL is set, loading only local deployments")
        else:
            # Otherwise, load non-local deployments (or deployments where local doesn't exist)
            deployment_filter = {
                "platform": ClientType.DISCORD.value,
                "$or": [{"local": {"$exists": False}}, {"local": {"$ne": True}}],
            }
            logger.info("Loading non-local deployments")

        # Add user filter if LOCAL_USER_ID is set
        if local_user_id:
            try:
                user_oid = BsonObjectId(local_user_id)
                deployment_filter["user"] = user_oid
                logger.info(f"Filtering deployments for user: {local_user_id}")
            except Exception as e:
                logger.error(
                    f"Invalid LOCAL_USER_ID format: {local_user_id}, error: {e}"
                )

        deployments = Deployment.find(deployment_filter)
        for deployment in deployments:
            # Skip deployments that are marked as invalid
            if deployment.valid is False:
                logger.info(f"Skipping invalid deployment {deployment.id}")
                continue

            if deployment.secrets and deployment.secrets.discord.token:
                await self.start_client(deployment)

        # Also load Telegram deployments for typing
        if local_api_url and local_api_url != "":
            telegram_filter = {"platform": ClientType.TELEGRAM.value, "local": True}
        else:
            telegram_filter = {
                "platform": ClientType.TELEGRAM.value,
                "$or": [{"local": {"$exists": False}}, {"local": {"$ne": True}}],
            }

        # Add user filter to Telegram deployments if LOCAL_USER_ID is set
        if local_user_id:
            try:
                user_oid = BsonObjectId(local_user_id)
                telegram_filter["user"] = user_oid
            except Exception as e:
                logger.error(
                    f"Invalid LOCAL_USER_ID format for Telegram filter: {local_user_id}, error: {e}"
                )

        telegram_deployments = Deployment.find(telegram_filter)
        for deployment in telegram_deployments:
            if (
                deployment.secrets
                and deployment.secrets.telegram
                and deployment.secrets.telegram.token
            ):
                self.telegram_typing_manager.register_deployment(
                    str(deployment.id), deployment.secrets.telegram.token
                )
                logger.info(
                    f"Registered Telegram deployment {deployment.id} for typing"
                )

    async def start_client(self, deployment: Deployment):
        """Start a new gateway client for a deployment"""
        deployment_id = str(deployment.id)
        if deployment_id in self.clients:
            logger.info(f"Gateway client for deployment {deployment_id} already exists")
            return

        # Check if the deployment is marked as invalid
        if deployment.valid is False:
            logger.info(f"Skipping invalid deployment {deployment_id}")
            return

        client = DiscordGatewayClient(deployment)
        self.clients[deployment_id] = client
        asyncio.create_task(client.connect())
        logger.info(f"Started gateway client for deployment {deployment_id}")

    async def stop_client(self, deployment_id: str):
        """Stop a gateway client"""
        if deployment_id in self.clients:
            client = self.clients.pop(deployment_id)  # Remove from dict first
            client.stop()
            logger.info(f"Stopped gateway client for deployment {deployment_id}")
        else:
            logger.info(f"No gateway client found for deployment {deployment_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the gateway manager
    manager = GatewayManager()

    # Set up Ably and load deployments
    await manager.setup_ably()
    await manager.load_deployments()

    # Store the manager in app state for access in routes if needed
    app.state.manager = manager

    yield

    # Clean up on shutdown
    logger.info("Shutting down gateway manager")
    # Stop all clients
    stop_tasks = []
    for deployment_id in list(manager.clients.keys()):
        stop_tasks.append(manager.stop_client(deployment_id))

    if stop_tasks:
        await asyncio.gather(*stop_tasks, return_exceptions=True)

    # Clean up Ably connections
    try:
        if manager.ably_client:
            await manager.ably_client.close()
    except Exception as e:
        logger.error(f"Error closing Ably client: {e}")

    # Give tasks a moment to clean up
    await asyncio.sleep(0.5)


web_app = FastAPI(lifespan=lifespan)


@web_app.get("/telegram/webhook")
async def telegram_webhook_validate():
    """Handle Telegram webhook validation requests"""
    return {"ok": True}


@web_app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request, x_telegram_bot_api_secret_token: str = Header(None)
):
    """Handle incoming Telegram webhook updates"""
    try:
        # Parse request body to check if it's a validation request
        try:
            body = await request.json()
        except Exception:
            # Empty body - likely validation request
            logger.info("Received POST with no body - validation request")
            return {"ok": True}

        # If body is empty dict, it's also a validation request
        if not body:
            logger.info("Received POST with empty body - validation request")
            return {"ok": True}

        # Find deployment by matching webhook secret
        deployment = None
        for dep in Deployment.find(
            {"platform": ClientType.TELEGRAM.value, "valid": {"$ne": False}}
        ):
            if (
                dep.secrets
                and dep.secrets.telegram
                and dep.secrets.telegram.webhook_secret
                == x_telegram_bot_api_secret_token
            ):
                deployment = dep
                break

        if not deployment:
            # During webhook setup, Telegram may send test requests without secret token
            # Return 200 OK to allow webhook validation to succeed
            logger.info(
                "No deployment found for webhook secret - returning OK for validation"
            )
            return {"ok": True}

        # Instantiate TelegramClient and call interact
        from eve.agent.deployments.telegram import TelegramClient

        telegram_client = TelegramClient(deployment=deployment)
        await telegram_client.interact(request)

        return {"ok": True}

    except Exception as e:
        logger.error(f"Error handling Telegram webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _refresh_gmail_watches(
    deployment_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    deployments: List[Deployment] = []
    results: List[Dict[str, Any]] = []

    if deployment_ids:
        for dep_id in deployment_ids:
            try:
                deployment = Deployment.from_mongo(ObjectId(dep_id))
            except Exception as exc:
                results.append(
                    {
                        "deployment_id": str(dep_id),
                        "status": "error",
                        "error": f"Deployment lookup failed: {exc}",
                    }
                )
                continue

            if not deployment or not _is_gmail_platform(deployment.platform):
                results.append(
                    {
                        "deployment_id": str(dep_id),
                        "status": "skipped",
                        "reason": "Not a Gmail deployment",
                    }
                )
                continue

            deployments.append(deployment)
    else:
        deployments = list(
            Deployment.find({"platform": ClientType.GMAIL.value, "valid": {"$ne": False}})
        )

    for deployment in deployments:
        dep_id = str(deployment.id)
        try:
            agent = Agent.from_mongo(ObjectId(deployment.agent))
        except Exception as exc:
            results.append(
                {
                    "deployment_id": dep_id,
                    "status": "error",
                    "error": f"Agent lookup failed: {exc}",
                }
            )
            continue

        if not agent:
            results.append(
                {
                    "deployment_id": dep_id,
                    "status": "error",
                    "error": "Agent not found",
                }
            )
            continue

        client = GmailClient(agent=agent, deployment=deployment)
        try:
            response = await client.ensure_watch()
            results.append(
                {
                    "deployment_id": dep_id,
                    "status": "refreshed",
                    "expiration": response.get("expiration") if response else None,
                    "historyId": response.get("historyId") if response else None,
                }
            )
        except Exception as exc:
            logger.error(
                f"[GMAIL-WATCH] Failed to refresh watch for deployment {dep_id}: {exc}",
                exc_info=True,
            )
            results.append(
                {
                    "deployment_id": dep_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return results


@web_app.post("/gmail/webhook")
async def gmail_webhook(request: Request):
    """Handle inbound Gmail notifications delivered via Pub/Sub push."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        inner_payload, attributes = unwrap_pubsub_message(body)
    except Exception as e:
        logger.error(f"[GMAIL-WEBHOOK] Failed to decode payload: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Failed to decode Pub/Sub payload")

    deployment_id = (
        request.query_params.get("deployment_id")
        or inner_payload.get("deployment_id")
        or (attributes or {}).get("deployment_id")
    )
    if not deployment_id:
        raise HTTPException(status_code=400, detail="deployment_id missing from payload")

    try:
        deployment = Deployment.from_mongo(ObjectId(deployment_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if not deployment or not _is_gmail_platform(deployment.platform):
        raise HTTPException(status_code=404, detail="Gmail deployment not found")

    agent = Agent.from_mongo(ObjectId(deployment.agent))
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found for deployment")

    gmail_client = GmailClient(agent=agent, deployment=deployment)

    history_id = inner_payload.get("historyId") or inner_payload.get("history_id")
    if history_id and not inner_payload.get("message_id"):
        try:
            result = await gmail_client.process_history_update(str(history_id))
            return {"status": "history_processed", "result": result}
        except APIError as exc:
            logger.error(f"[GMAIL-WEBHOOK] History processing API error: {exc}")
            raise HTTPException(status_code=exc.status_code or 500, detail=str(exc))
        except Exception as exc:
            logger.error(
                f"[GMAIL-WEBHOOK] Failed to process history update: {exc}",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Failed to process history update")

    try:
        email = parse_inbound_email(inner_payload)
    except ValueError as exc:
        logger.warning(f"[GMAIL-WEBHOOK] Ignoring payload: {exc}")
        return {"status": "ignored", "reason": str(exc)}
    except Exception as exc:
        logger.error(f"[GMAIL-WEBHOOK] Failed to parse email payload: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid email payload")

    try:
        await gmail_client.process_inbound_email(email)
    except APIError as exc:
        logger.error(f"[GMAIL-WEBHOOK] Processing failed with API error: {exc}")
        raise HTTPException(status_code=exc.status_code or 500, detail=str(exc))
    except Exception as exc:
        logger.error(f"[GMAIL-WEBHOOK] Error processing inbound email: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process email")

    return {"status": "processed"}



@web_app.post("/gmail/watch/refresh")
async def gmail_watch_refresh(request: Request):
    """Manually refresh Gmail watches for one or more deployments."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    deployment_ids = body.get("deployment_ids") or body.get("deployment_id")
    if isinstance(deployment_ids, str):
        deployment_ids = [deployment_ids]
    elif deployment_ids is not None and not isinstance(deployment_ids, list):
        deployment_ids = [str(deployment_ids)]

    results = await _refresh_gmail_watches(deployment_ids)
    return {"results": results}


if WATCH_REFRESH_INTERVAL_HOURS > 0:

    @app.function(
        image=image,
        schedule=modal.Period(hours=WATCH_REFRESH_INTERVAL_HOURS),
        timeout=600,
    )
    async def gmail_watch_refresher():
        logger.info(
            f"[GMAIL-WATCH] Periodic refresh triggered (interval={WATCH_REFRESH_INTERVAL_HOURS}h)"
        )
        await _refresh_gmail_watches()


@app.function(
    image=image,
    min_containers=1,
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gateway_app():
    return web_app
