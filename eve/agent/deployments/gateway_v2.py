from pathlib import Path
import modal
import asyncio
import json
import logging
import os
from typing import Dict, Optional, Tuple
import websockets
import aiohttp
from ably import AblyRealtime

from eve import db
from eve.user import User
from eve.agent.session.models import (
    ChatMessageRequestInput,
    Session,
    SessionUpdateConfig,
    Deployment,
    ClientType,
)
from eve.api.api_requests import SessionCreationArgs, PromptSessionRequest
import eve.mongo
from fastapi import FastAPI
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
root_dir = Path(__file__).parent.parent.parent.parent

# Create Modal app
app = modal.App(
    name=f"discord-gateway-v2-{db}",
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
    .add_local_python_source("eve", ignore=[])
)


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


class TypingManager:
    """Manages typing indicators for Discord channels"""

    def __init__(self, client):
        self.client = client
        # Store task and last keepalive timestamp
        self.typing_channels: Dict[str, Dict] = {}
        self.check_interval = 60  # Check for stale tasks every 60 seconds
        self.stale_threshold = (
            300  # Consider typing stale after 5 minutes without keepalive
        )
        self._check_task: Optional[asyncio.Task] = None  # Store check task

    def start_check_loop(self):
        """Starts the background loop to check for stale typing tasks."""
        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._check_stale_loop())
            logger.info("Started TypingManager stale check loop.")

    async def _check_stale_loop(self):
        """Periodically check for and clean up stale typing tasks."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                now = asyncio.get_event_loop().time()
                stale_channels = []
                for channel_id, state in list(self.typing_channels.items()):
                    if isinstance(state, dict) and "last_keepalive" in state:
                        if now - state.get("last_keepalive", 0) > self.stale_threshold:
                            stale_channels.append(channel_id)
                            # Log the specific channel being marked as stale
                            logger.warning(
                                f"[StaleCheck] Marking channel {channel_id} as stale (last keepalive: {state.get('last_keepalive')}, now: {now})."
                            )
                    else:
                        logger.warning(
                            f"[StaleCheck] Found invalid typing state for channel {channel_id}, marking stale: {state}"
                        )
                        stale_channels.append(channel_id)

                if stale_channels:
                    logger.warning(
                        f"[StaleCheck] Attempting to stop stale typing indicators for channels: {stale_channels}."
                    )
                    await asyncio.gather(
                        *[
                            self.stop_typing(channel_id, reason="stale")
                            for channel_id in stale_channels
                        ],
                        return_exceptions=True,
                    )

            except asyncio.CancelledError:
                logger.info("[StaleCheck] TypingManager stale check loop cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"[StaleCheck] Error in TypingManager stale check loop: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(self.check_interval)

    async def start_typing(self, channel_id: str):
        """Start typing in a channel or update keepalive if already typing."""
        now = asyncio.get_event_loop().time()
        state = self.typing_channels.get(channel_id)

        if state and isinstance(state, dict):
            # Update keepalive timestamp
            state["last_keepalive"] = now
            task = state.get("task")
            # Check if task exists and is running
            if task and isinstance(task, asyncio.Task) and not task.done():
                logger.debug(f"Updated keepalive for typing in channel {channel_id}")
                return  # Already typing and keepalive updated
            else:
                logger.warning(
                    f"Found existing state but task invalid/done for channel {channel_id}, restarting task."
                )
                # Ensure old task is cancelled if it exists but is done/invalid
                if task and isinstance(task, asyncio.Task) and not task.cancelled():
                    task.cancel()

        # If state doesn't exist or task needs restarting
        logger.info(f"Starting typing task for channel {channel_id}")
        task = asyncio.create_task(self._typing_loop(channel_id))
        self.typing_channels[channel_id] = {"task": task, "last_keepalive": now}
        # Ensure the check loop is running
        self.start_check_loop()

    async def stop_typing(self, channel_id: str, reason: str = "signal"):
        """Stop typing in a channel."""
        logger.info(
            f"[StopTyping:{reason}] Attempting to stop typing for channel {channel_id}."
        )
        state = self.typing_channels.pop(channel_id, None)
        if state and isinstance(state, dict):
            task = state.get("task")
            if task and isinstance(task, asyncio.Task) and not task.done():
                logger.info(
                    f"[StopTyping:{reason}] Found active task for channel {channel_id}. Cancelling..."
                )
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                    logger.info(
                        f"[StopTyping:{reason}] Typing task cancellation confirmed for channel {channel_id}."
                    )
                except asyncio.CancelledError:
                    logger.info(
                        f"[StopTyping:{reason}] Typing task cancellation confirmed (exception) for channel {channel_id}."
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[StopTyping:{reason}] Timeout waiting for typing task cancellation for channel {channel_id}."
                    )
                except Exception as e:
                    logger.error(
                        f"[StopTyping:{reason}] Error awaiting cancelled typing task for channel {channel_id}: {e}",
                        exc_info=True,
                    )
            elif task:
                logger.debug(
                    f"[StopTyping:{reason}] Typing task for channel {channel_id} already done or invalid, removed state."
                )
            else:
                logger.warning(
                    f"[StopTyping:{reason}] State existed for channel {channel_id} but task was missing or invalid."
                )
            logger.info(
                f"[StopTyping:{reason}] Removed state for channel {channel_id}. Current states: {list(self.typing_channels.keys())}"
            )
        else:
            logger.debug(
                f"[StopTyping:{reason}] No active typing state found to stop for channel {channel_id}."
            )

    async def _typing_loop(self, channel_id: str):
        """Loop that sends typing indicators every ~8 seconds."""
        typing_interval = 8  # Discord times out typing after 10 seconds
        try:
            while True:
                # Check if state still exists before updating keepalive/sending
                current_state = self.typing_channels.get(channel_id)
                if not current_state or not isinstance(current_state, dict):
                    logger.warning(
                        f"Typing loop for channel {channel_id} found state missing/invalid, stopping loop."
                    )
                    break

                # Update keepalive timestamp
                current_state["last_keepalive"] = asyncio.get_event_loop().time()
                logger.debug(
                    f"Sending typing indicator and updated keepalive for {channel_id}"
                )

                await self._send_typing_indicator(channel_id)
                await asyncio.sleep(typing_interval)

        except asyncio.CancelledError:
            # Expected when stop_typing is called
            logger.info(f"Typing loop gracefully cancelled for channel {channel_id}")
            # Do not re-raise, allow loop to exit cleanly
        except Exception as e:
            logger.error(
                f"Error in typing loop for channel {channel_id}: {e}", exc_info=True
            )
        finally:
            # Log when this loop actually terminates
            logger.info(f"[_TypingLoop] Exiting typing loop for channel {channel_id}.")
            # Ensure channel is removed from tracking if loop exits unexpectedly
            if channel_id in self.typing_channels:
                logger.warning(
                    f"[_TypingLoop] Typing loop for {channel_id} ended unexpectedly (state still existed), cleaning up state."
                )
                # Call stop_typing to ensure proper cleanup, even if task is already cancelled
                await self.stop_typing(channel_id, reason="loop_exit")

    async def _send_typing_indicator(self, channel_id: str):
        """Send a typing indicator to a Discord channel."""
        try:
            # Use the client's session if available, otherwise create one
            # TODO: Refactor client to hold a persistent aiohttp session?
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bot {self.client.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "EveDiscordClient (https://github.com/your-repo, 1.0)",  # Good practice
                }
                url = f"https://discord.com/api/v10/channels/{channel_id}/typing"
                async with session.post(url, headers=headers) as response:
                    if response.status == 204:
                        logger.debug(
                            f"Successfully sent typing indicator to {channel_id}"
                        )
                    elif response.status == 429:  # Rate limited
                        retry_after = float(
                            await response.json().get("retry_after", 1.0)
                        )
                        logger.warning(
                            f"Rate limited sending typing to {channel_id}, retrying after {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                    elif response.status in [401, 403]:  # Permissions issue
                        logger.error(
                            f"Permissions error sending typing to {channel_id} (Status: {response.status}). Stopping typing for this channel."
                        )
                        # Stop the loop by removing the channel state
                        self.typing_channels.pop(channel_id, None)
                    elif response.status == 404:  # Channel not found
                        logger.error(
                            f"Channel {channel_id} not found sending typing. Stopping typing for this channel."
                        )
                        self.typing_channels.pop(channel_id, None)
                    else:
                        logger.warning(
                            f"Failed to send typing indicator to {channel_id}: {response.status} - {await response.text()}"
                        )
        except aiohttp.ClientError as e:
            logger.error(
                f"HTTP Client error sending typing indicator to {channel_id}: {e}"
            )
            # Consider temporary backoff?
            await asyncio.sleep(5)  # Basic retry delay
        except Exception as e:
            logger.error(
                f"Unexpected error sending typing indicator to {channel_id}: {e}",
                exc_info=True,
            )
            # Consider stopping the loop if error persists?
            await asyncio.sleep(5)  # Basic retry delay

    async def cleanup(self):
        """Cleans up all active typing tasks and the check loop."""
        logger.info("Cleaning up TypingManager tasks...")
        # Cancel the check loop first
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()

        # Stop all active typing loops
        active_channels = list(self.typing_channels.keys())
        if active_channels:
            logger.info(f"Stopping active typing for channels: {active_channels}")
            await asyncio.gather(
                *[self.stop_typing(channel_id) for channel_id in active_channels],
                return_exceptions=True,
            )

        # Wait for the check loop task to finish cancellation
        if self._check_task:
            try:
                await asyncio.wait_for(self._check_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.info("TypingManager check loop cancellation confirmed.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for typing manager check loop cancellation."
                )
            except Exception as e:
                logger.error(
                    f"Error awaiting check loop cancellation: {e}", exc_info=True
                )

        self.typing_channels.clear()  # Ensure dict is empty
        logger.info("TypingManager cleanup complete.")


class TelegramTypingManager:
    """Manages typing indicators for Telegram chats"""

    def __init__(self):
        self.typing_chats = {}  # chat_id_thread_id -> typing task
        self.tokens = {}  # deployment_id -> bot token

    def register_deployment(self, deployment_id: str, token: str):
        """Register a new deployment with its token"""
        self.tokens[deployment_id] = token
        logger.info(f"Registered Telegram deployment {deployment_id} for typing")

    def unregister_deployment(self, deployment_id: str):
        """Unregister a deployment"""
        if deployment_id in self.tokens:
            del self.tokens[deployment_id]
            logger.info(f"Unregistered Telegram deployment {deployment_id} from typing")

            # Stop any active typing for this deployment
            # This is a safety measure to ensure no lingering typing indicators
            chats_to_stop = []
            for chat_key in self.typing_chats:
                if chat_key.startswith(f"{deployment_id}:"):
                    chats_to_stop.append(chat_key)

            for chat_key in chats_to_stop:
                task = self.typing_chats.pop(chat_key, None)
                if task and not task.done():
                    task.cancel()

    async def start_typing(self, deployment_id, chat_id, thread_id=None):
        """Start typing in a Telegram chat"""
        if deployment_id not in self.tokens:
            logger.warning(f"No token found for deployment {deployment_id}")
            return

        chat_key = f"{chat_id}_{thread_id}" if thread_id else str(chat_id)

        if chat_key in self.typing_chats and not self.typing_chats[chat_key].done():
            return  # Already typing in this chat

        self.typing_chats[chat_key] = asyncio.create_task(
            self._typing_loop(deployment_id, chat_id, thread_id)
        )
        logger.info(f"Started typing in Telegram chat {chat_key}")

    async def stop_typing(self, chat_id, thread_id=None):
        """Stop typing in a Telegram chat"""
        chat_key = f"{chat_id}_{thread_id}" if thread_id else str(chat_id)
        task = self.typing_chats.pop(chat_key, None)
        if task and not task.done():
            task.cancel()
            logger.info(f"Stopped typing in Telegram chat {chat_key}")

    async def _typing_loop(self, deployment_id, chat_id, thread_id):
        """Loop that sends typing indicators every 5 seconds"""
        try:
            token = self.tokens.get(deployment_id)
            if not token:
                return

            while True:
                await self._send_typing_indicator(token, chat_id, thread_id)
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info(f"Typing cancelled for Telegram chat {chat_id}")
        except Exception as e:
            logger.error(f"Error in Telegram typing loop: {e}")

    async def _send_typing_indicator(self, token, chat_id, thread_id):
        """Send a typing indicator to a Telegram chat"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{token}/sendChatAction"
                payload = {"chat_id": chat_id, "action": "typing"}
                if thread_id:
                    payload["message_thread_id"] = thread_id

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to send Telegram typing indicator: {await response.text()}"
                        )
        except Exception as e:
            logger.error(f"Error sending Telegram typing indicator: {e}")


class DiscordGatewayClient:
    GATEWAY_VERSION = 10
    GATEWAY_URL = f"wss://gateway.discord.gg/?v={GATEWAY_VERSION}&encoding=json"

    def __init__(self, deployment: Deployment, manager=None):
        if deployment.platform != ClientType.DISCORD:
            raise ValueError("Deployment must be for Discord HTTP platform")

        self.deployment = deployment
        self.manager = manager  # Reference to GatewayManager for in-memory cache access
        self.token = deployment.secrets.discord.token
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.heartbeat_interval: Optional[int] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect = True
        self._last_sequence = None
        self._session_id = None
        self.typing_manager = TypingManager(self)

        # Set up Ably for busy state updates
        self.ably_client = None
        self.busy_channel = None

    @property
    def api_url(self) -> str:
        """Get the API URL, preferring LOCAL_API_URL if set."""
        if os.getenv("LOCAL_API_URL") != "":
            return os.getenv("LOCAL_API_URL")
        else:
            return os.getenv("EDEN_API_URL")

    async def heartbeat_loop(self):
        while True:
            await self.ws.send(
                json.dumps({"op": GatewayOpCode.HEARTBEAT, "d": self._last_sequence})
            )
            await asyncio.sleep(self.heartbeat_interval / 1000)

    async def identify(self):
        logger.info(f"Identifying for deployment {self.deployment.id}")
        await self.ws.send(
            json.dumps(
                {
                    "op": GatewayOpCode.IDENTIFY,
                    "d": {
                        "token": self.token,
                        "intents": 1 << 9 | 1 << 15,  # GUILD_MESSAGES | MESSAGE_CONTENT
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
        Uses in-memory cache instead of database queries for better performance.
        :param message_data: Discord message data containing mentions and mention_roles
        :return: List of agent IDs that were mentioned
        """
        mentioned_agent_ids = []

        # If no manager reference, fall back to empty list (shouldn't happen)
        if not self.manager:
            logger.warning("No manager reference available for mention parsing")
            return mentioned_agent_ids

        # Check regular user mentions
        mentions = message_data.get("mentions", [])
        for mention in mentions:
            discord_id = mention.get("id")
            if discord_id and discord_id in self.manager.discord_app_id_to_agent_id:
                agent_id = self.manager.discord_app_id_to_agent_id[discord_id]
                if agent_id not in mentioned_agent_ids:
                    mentioned_agent_ids.append(agent_id)
                    logger.info(f"Found mentioned agent via user mention: {agent_id}")

        # Check role mentions (this is where your @&1367139358905471029 would be)
        mention_roles = message_data.get("mention_roles", [])
        for role_id in mention_roles:
            if role_id and role_id in self.manager.discord_app_id_to_agent_id:
                agent_id = self.manager.discord_app_id_to_agent_id[role_id]
                if agent_id not in mentioned_agent_ids:
                    mentioned_agent_ids.append(agent_id)
                    logger.info(f"Found mentioned agent via role mention: {agent_id}")

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

    def _extract_user_info(self, message_data: dict) -> Tuple[str, str, User]:
        """Extract user information from message data"""
        user_id = message_data.get("author", {}).get("id")
        username = message_data.get("author", {}).get("username", "User")
        user = User.from_discord(user_id, username)
        return user_id, username, user

    def _process_message_content(
        self, message_data: dict
    ) -> Tuple[str, list, bool, list]:
        """Process message content, mentions, and attachments"""
        logger.info("Processing message content", extra={"message_data": message_data})
        content = message_data["content"]
        mentioned_agent_ids = []
        print(f"message_data: {message_data}")

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

        # Get attachments
        attachments = []
        if "attachments" in message_data:
            attachments = [
                attachment.get("proxy_url")
                for attachment in message_data["attachments"]
                if "proxy_url" in attachment
            ]

        # Check if bot is mentioned (force reply) - check both mentions and mention_roles
        force_reply = False
        app_id = self.deployment.secrets.discord.application_id

        # Check user mentions
        if message_data.get("mentions") and any(
            mention.get("id") == app_id for mention in message_data.get("mentions", [])
        ):
            force_reply = True

        # Check role mentions
        if not force_reply and app_id in message_data.get("mention_roles", []):
            force_reply = True

        content = content or "..."
        return content, attachments, force_reply, mentioned_agent_ids

    def _create_session_key(self, message_data: dict, user_id: str) -> str:
        """Create a session key based on Discord channel"""
        is_dm = message_data.get("guild_id") is None
        if is_dm:
            return f"discord-dm-{user_id}"
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
            logger.info(f"[{trace_id}] Found existing session: {session.id}")
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

    def _create_session_request(
        self,
        user: User,
        username: str,
        content: str,
        attachments: list,
        channel_id: str,
        message_id: str,
        mentioned_agent_ids: list = None,
    ) -> PromptSessionRequest:
        """Create a PromptSessionRequest object"""
        # If specific agents are mentioned, use those; otherwise use this deployment's agent
        print(f"mentioned_agent_ids: {mentioned_agent_ids}")

        if mentioned_agent_ids:
            actor_agent_ids = mentioned_agent_ids
        else:
            actor_agent_ids = []
        print(f"actor_agent_ids: {actor_agent_ids}")

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
                update_endpoint=f"{self.api_url}/v2/deployments/emission",
                discord_channel_id=channel_id,
                discord_message_id=message_id,
            ),
        )

    async def _send_session_request(
        self, request: PromptSessionRequest, trace_id: str
    ) -> None:
        """Send the session request to the API"""
        logger.info(
            f"[{trace_id}] Sending session request to {self.api_url}/sessions/prompt"
        )
        logger.debug(f"[{trace_id}] Request payload: {request.model_dump()}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/sessions/prompt",
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

        channel_id = str(data["channel_id"])
        if not await self._check_channel_permissions(channel_id, deployment, trace_id):
            return

        # Extract message information
        user_id, username, user = self._extract_user_info(data)
        content, attachments, force_reply, mentioned_agent_ids = (
            self._process_message_content(data)
        )
        logger.info(f"[{trace_id}] mentioned_agent_ids: {mentioned_agent_ids}")
        session_key = self._create_session_key(data, user_id)

        logger.debug(
            f"[{trace_id}] Session key: {session_key}, force_reply: {force_reply}, mentioned_agents: {mentioned_agent_ids}"
        )

        # Check for agent conflicts before proceeding
        relevant_deployments = await self._find_relevant_deployments(
            channel_id, trace_id
        )
        if self._should_skip_for_agent_conflict(
            relevant_deployments, force_reply, trace_id
        ):
            return

        # Handle session creation/reuse
        session = await self._load_existing_session(session_key, trace_id)
        request = self._create_session_request(
            user,
            username,
            content,
            attachments,
            channel_id,
            str(data["id"]),
            mentioned_agent_ids,
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
                    is_busy = data.get(
                        "is_busy"
                    )  # Don't default to False, check existence

                    # Log exactly what was received
                    logger.info(
                        f"[{trace_id}] Ably message received: channel_id={channel_id}, is_busy={is_busy} (type: {type(is_busy)}), data={data}"
                    )

                    if channel_id is None:
                        logger.warning(
                            f"[{trace_id}] Received busy state update without channel_id: {data}"
                        )
                        return

                    # Only proceed if is_busy is explicitly True or False
                    if is_busy is True:
                        channel_id_str = str(channel_id)
                        logger.info(
                            f"[{trace_id}] Calling start_typing for channel {channel_id_str}"
                        )
                        await self.typing_manager.start_typing(channel_id_str)
                    elif is_busy is False:
                        channel_id_str = str(channel_id)
                        logger.info(
                            f"[{trace_id}] Received is_busy=False. Calling stop_typing for channel {channel_id_str}"
                        )
                        await self.typing_manager.stop_typing(
                            channel_id_str, reason="ably_signal"
                        )
                    else:
                        logger.warning(
                            f"[{trace_id}] Received invalid or missing 'is_busy' value ({is_busy}). Ignoring message: {data}"
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
        # Start the typing manager's check loop
        self.typing_manager.start_check_loop()

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
                            close_code = data.get("d", {}).get("code", 0)
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

            except websockets.exceptions.ConnectionClosedError as e:
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
                if self.ws and self.ws.open:
                    await self.ws.close()
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
        await self.typing_manager.cleanup()
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
        if self.ws and self.ws.open:
            logger.info(f"Closing WebSocket connection for {self.deployment.id}")
            # Schedule the close, don't block here
            asyncio.create_task(self.ws.close(code=1000, reason="Client stopping"))
        else:
            # If WS not open/already closed, manually trigger cleanup if needed
            # Check if loop is running to avoid errors
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logger.info(
                        f"WS not open for {self.deployment.id}, scheduling resource cleanup."
                    )
                    asyncio.create_task(self.cleanup_resources())
                else:
                    # Not ideal to run_until_complete here, but as fallback
                    logger.info(
                        f"WS not open and loop not running for {self.deployment.id}, running cleanup synchronously."
                    )
                    loop.run_until_complete(self.cleanup_resources())
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

        # Add Telegram typing manager
        self.telegram_typing_manager = TelegramTypingManager()

        # In-memory cache mapping Discord application IDs to agent IDs
        self.discord_app_id_to_agent_id: Dict[str, str] = {}

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

            # Update the in-memory cache
            if (
                deployment.secrets
                and deployment.secrets.discord
                and deployment.secrets.discord.application_id
            ):
                app_id = deployment.secrets.discord.application_id
                agent_id = str(deployment.agent)
                self.discord_app_id_to_agent_id[app_id] = agent_id
                logger.info(
                    f"[trace:{reload_trace_id}] Updated cache: Discord app ID {app_id} -> agent ID {agent_id}"
                )

            # Create a completely new client with the fresh data
            client = DiscordGatewayClient(deployment, manager=self)
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
                    is_busy = data.get("is_busy", False)

                    if deployment_id and chat_id:
                        if is_busy:
                            await self.telegram_typing_manager.start_typing(
                                deployment_id, chat_id, thread_id
                            )
                        else:
                            await self.telegram_typing_manager.stop_typing(
                                chat_id, thread_id
                            )
                except Exception as e:
                    logger.error(f"Error handling Telegram busy state update: {e}")

            await telegram_channel.subscribe(telegram_message_handler)
            logger.info("Subscribed to Telegram busy state updates")

        except Exception as e:
            logger.error(f"Failed to setup Telegram Ably subscription: {e}")

    async def load_deployments(self):
        """Load all Discord HTTP deployments from database"""
        deployments = Deployment.find({"platform": ClientType.DISCORD.value})
        for deployment in deployments:
            # Skip deployments that are marked as invalid
            if deployment.valid is False:
                logger.info(f"Skipping invalid deployment {deployment.id}")
                continue

            if deployment.secrets and deployment.secrets.discord.token:
                await self.start_client(deployment)

        # Also load Telegram deployments for typing
        telegram_deployments = Deployment.find({"platform": ClientType.TELEGRAM.value})
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

        # Add to the in-memory cache
        if (
            deployment.secrets
            and deployment.secrets.discord
            and deployment.secrets.discord.application_id
        ):
            app_id = deployment.secrets.discord.application_id
            agent_id = str(deployment.agent)
            self.discord_app_id_to_agent_id[app_id] = agent_id
            logger.info(f"Cached Discord app ID {app_id} -> agent ID {agent_id}")

        client = DiscordGatewayClient(deployment, manager=self)
        self.clients[deployment_id] = client
        asyncio.create_task(client.connect())
        logger.info(f"Started gateway client for deployment {deployment_id}")

    async def stop_client(self, deployment_id: str):
        """Stop a gateway client"""
        if deployment_id in self.clients:
            client = self.clients.pop(deployment_id)  # Remove from dict first

            # Clean up the cache entry
            if (
                client.deployment.secrets
                and client.deployment.secrets.discord
                and client.deployment.secrets.discord.application_id
            ):
                app_id = client.deployment.secrets.discord.application_id
                if app_id in self.discord_app_id_to_agent_id:
                    del self.discord_app_id_to_agent_id[app_id]
                    logger.info(f"Removed Discord app ID {app_id} from cache")

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
    for deployment_id in list(manager.clients.keys()):
        await manager.stop_client(deployment_id)


web_app = FastAPI(lifespan=lifespan)


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
