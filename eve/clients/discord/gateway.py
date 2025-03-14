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
from eve.deploy import Deployment, ClientType
from eve.user import User
from eve.agent import Agent
from fastapi import FastAPI
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
root_dir = Path(__file__).parent.parent.parent.parent

# Create Modal app
app = modal.App(
    name=f"discord-gateway-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
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
        self.typing_channels = {}  # channel_id -> typing task

    async def start_typing(self, channel_id):
        """Start typing in a channel"""
        if (
            channel_id in self.typing_channels
            and not self.typing_channels[channel_id].done()
        ):
            return  # Already typing in this channel

        self.typing_channels[channel_id] = asyncio.create_task(
            self._typing_loop(channel_id)
        )
        logger.info(f"Started typing in channel {channel_id}")

    async def stop_typing(self, channel_id):
        """Stop typing in a channel"""
        task = self.typing_channels.pop(channel_id, None)
        if task and not task.done():
            task.cancel()
            logger.info(f"Stopped typing in channel {channel_id}")

    async def _typing_loop(self, channel_id):
        """Loop that sends typing indicators every 5 seconds"""
        try:
            while True:
                await self._send_typing_indicator(channel_id)
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info(f"Typing cancelled for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error in typing loop: {e}")

    async def _send_typing_indicator(self, channel_id):
        """Send a typing indicator to a Discord channel"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bot {self.client.token}",
                    "Content-Type": "application/json",
                }

                url = f"https://discord.com/api/v10/channels/{channel_id}/typing"
                async with session.post(url, headers=headers) as response:
                    if response.status != 204:
                        logger.warning(
                            f"Failed to send typing indicator: {await response.text()}"
                        )
        except Exception as e:
            logger.error(f"Error sending typing indicator: {e}")


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
        self.typing_manager = TypingManager(self)

        # Set up Ably for busy state updates
        self.ably_client = None
        self.busy_channel = None

        # Add dictionaries to track known users and threads
        self.known_users = {}
        self.known_threads = {}

    async def heartbeat_loop(self):
        while True:
            await self.ws.send(
                json.dumps({"op": GatewayOpCode.HEARTBEAT, "d": self._last_sequence})
            )
            logger.info(
                f"Sent heartbeat for deployment {self.deployment.id} with interval {self.heartbeat_interval}"
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

    def lookup_thread_and_user(self, data: dict) -> Tuple[str, str]:
        """
        Helper function to lookup thread and user for a Discord message.

        Args:
            data: The Discord message data

        Returns:
            A tuple containing (thread, user) objects
        """
        # Get user ID and name from message data
        user_id = data.get("author", {}).get("id")
        username = data.get("author", {}).get("username", "User")

        # Determine if this is a DM or guild message
        is_dm = data.get("guild_id") is None
        channel_id = data.get("channel_id")

        # Generate the thread key
        if is_dm:
            thread_key = f"discord-dm-{username}-{user_id}"
        else:
            guild_id = data.get("guild_id")
            thread_key = f"discord-{guild_id}-{channel_id}"

        # Lookup thread
        if thread_key not in self.known_threads:
            agent = Agent.from_mongo(str(self.deployment.agent))
            self.known_threads[thread_key] = agent.request_thread(key=thread_key)
        thread = self.known_threads[thread_key]

        # Lookup user
        if user_id not in self.known_users:
            self.known_users[user_id] = User.from_discord(user_id, username)
        user = self.known_users[user_id]

        return thread, user

    async def handle_message(self, data: dict):
        logger.info(
            f"Handling message for deployment {self.deployment.id} with data {data}"
        )

        # Skip messages from the bot itself
        print("AUTHOR ID", data.get("author", {}).get("id"))
        if (
            data.get("author", {}).get("id")
            == self.deployment.secrets.discord.application_id
        ):
            print("SKIPPING MESSAGE FROM BOT")
            print("DEPLOYMENT ID", self.deployment.id)
            print("APPLICATION ID", self.deployment.secrets.discord.application_id)
            return

        # Fetch fresh deployment data to get the latest allowlist
        fresh_deployment = Deployment.from_mongo(str(self.deployment.id))
        if not fresh_deployment or not fresh_deployment.config:
            logger.info(f"No config found for deployment {self.deployment.id}")
            return

        channel_id = str(data["channel_id"])

        # Check against the freshly fetched allowlist
        if fresh_deployment.config.discord.channel_allowlist:
            allowed_channels = [
                item.id for item in fresh_deployment.config.discord.channel_allowlist
            ]
            print("ALLOWED CHANNELS", allowed_channels)
            print("CHANNEL ID", channel_id)

            if channel_id not in allowed_channels:
                print("NOT IN ALLOWED CHANNELS")
                return

        # Get thread and user using the helper function
        thread, user = self.lookup_thread_and_user(data)

        # Process message content
        content = data["content"]

        # Handle mentions
        if "mentions" in data:
            content = self.replace_mentions_with_usernames(content, data["mentions"])

        # Handle references/replies
        if data.get("referenced_message"):
            ref_message = data["referenced_message"]
            ref_content = ref_message.get("content", "")
            content = f"(Replying to message: {ref_content[:100]} ...)\n\n{content}"

        # Get attachments
        attachments = []
        if "attachments" in data:
            attachments = [
                attachment.get("proxy_url")
                for attachment in data["attachments"]
                if "proxy_url" in attachment
            ]

        # # Check if this is a direct mention or reply to the bot
        force_reply = False
        if data.get("mentions") and any(
            mention.get("id") == self.deployment.secrets.discord.application_id
            for mention in data.get("mentions", [])
        ):
            force_reply = True

        chat_request = {
            "agent_id": str(self.deployment.agent),
            "user_id": str(user.id),
            "thread_id": str(thread.id),
            "user_message": {
                "content": content,
                "role": "user",
                "name": data.get("author", {}).get("username", "User"),
                "attachments": attachments,
            },
            "update_config": {
                "deployment_id": str(self.deployment.id),
                "discord_channel_id": channel_id,
                "discord_message_id": str(data["id"]),
                "update_endpoint": f"{os.getenv('EDEN_API_URL')}/emissions/platform/discord",
            },
            "user_is_bot": data.get("author", {}).get("bot", False),
            "force_reply": force_reply,
        }

        print("CHAT REQUEST", chat_request)
        print("SENDING TO", f"{os.getenv('EDEN_API_URL')}/chat")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/chat",
                json=chat_request,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                    "X-Client-Platform": "discord",
                    "X-Client-Deployment-Id": str(self.deployment.id),
                },
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to process chat request: {await response.text()}"
                    )

    async def setup_ably(self):
        """Set up Ably for listening to busy state updates"""
        try:
            from ably import AblyRealtime

            self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
            channel_name = f"busy-state-discord-{self.deployment.id}"
            self.busy_channel = self.ably_client.channels.get(channel_name)

            async def message_handler(message):
                try:
                    data = message.data
                    if not isinstance(data, dict):
                        return

                    channel_id = data.get("channel_id")
                    is_busy = data.get("is_busy", False)

                    if channel_id:
                        if is_busy:
                            await self.typing_manager.start_typing(channel_id)
                        else:
                            await self.typing_manager.stop_typing(channel_id)
                            # Double-check after a short delay to ensure typing has stopped
                            await asyncio.sleep(0.5)
                            await self.typing_manager.stop_typing(channel_id)

                except Exception as e:
                    logger.error(f"Error handling busy state update: {e}")

            await self.busy_channel.subscribe(message_handler)
            logger.info(f"Subscribed to busy state updates: {channel_name}")

        except Exception as e:
            logger.error(f"Failed to setup Ably: {e}")

    async def connect(self):
        # Set up Ably for typing indicators
        await self.setup_ably()

        while self._reconnect:
            try:
                logger.info(
                    f"Connecting to gateway for deployment {self.deployment.id}"
                )
                async with websockets.connect(self.GATEWAY_URL) as ws:
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
                        logger.info(
                            f"Received message for deployment {self.deployment.id}"
                        )
                        data = json.loads(message)

                        if data.get("s"):
                            self._last_sequence = data["s"]

                        if data["op"] == GatewayOpCode.DISPATCH:
                            logger.info(
                                f"Dispatch event for deployment {self.deployment.id}"
                            )
                            if data["t"] == GatewayEvent.MESSAGE_CREATE:
                                logger.info(
                                    f"Handling message create event for deployment {self.deployment.id}"
                                )
                                await self.handle_message(data["d"])
                            elif data["t"] == GatewayEvent.READY:
                                logger.info(
                                    f"Ready event for deployment {self.deployment.id}"
                                )
                                self._session_id = data["d"]["session_id"]
                                logger.info(
                                    f"Gateway connected for deployment {self.deployment.id}"
                                )

            except Exception as e:
                logger.error(
                    f"Gateway connection error for deployment {self.deployment.id}: {e}"
                )
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                await asyncio.sleep(5)

    def stop(self):
        self._reconnect = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()


class GatewayManager:
    def __init__(self):
        self.clients: Dict[str, DiscordGatewayClient] = {}
        self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        self.channel = self.ably_client.channels.get(f"discord-gateway-{db}")

        # Add Telegram typing manager
        self.telegram_typing_manager = TelegramTypingManager()

        # Set up Ably for Telegram busy state updates
        self.telegram_busy_channel = None

    async def reload_client(self, deployment_id: str):
        """Reload a gateway client with fresh deployment data"""
        logger.info(f"Reloading gateway client for deployment {deployment_id}")

        # First stop and remove the existing client if it exists
        if deployment_id in self.clients:
            client = self.clients.pop(deployment_id)
            client.stop()
            logger.info(f"Stopped existing client for deployment {deployment_id}")

            # Add a small delay to ensure cleanup
            await asyncio.sleep(1)

        # Get fresh deployment data from database
        deployment = Deployment.from_mongo(deployment_id)
        if deployment:
            # Create a completely new client with the fresh data
            client = DiscordGatewayClient(deployment)
            self.clients[deployment_id] = client

            # Start the new client
            asyncio.create_task(client.connect())
            logger.info(
                f"Successfully reloaded gateway client for deployment {deployment_id} with fresh data"
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
                    f"Updated allowlist for deployment {deployment_id}: {allowed_channels}"
                )
        else:
            logger.error(f"Failed to reload - deployment not found: {deployment_id}")

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

                if not command or not deployment_id:
                    logger.warning(f"Missing command or deployment_id: {data}")
                    return

                logger.info(
                    f"Received command: {command} for deployment: {deployment_id}"
                )

                if command == "start":
                    # Start a new gateway client
                    deployment = Deployment.from_mongo(deployment_id)
                    if deployment:
                        await self.start_client(deployment)
                    else:
                        logger.error(f"Deployment not found: {deployment_id}")

                elif command == "stop":
                    # Stop an existing gateway client
                    await self.stop_client(deployment_id)

                # Add Telegram-specific commands
                elif command == "register_telegram":
                    # Register a new Telegram deployment
                    token = data.get("token")
                    if token:
                        self.telegram_typing_manager.register_deployment(
                            deployment_id, token
                        )
                    else:
                        logger.error(f"Missing token for Telegram registration: {data}")

                elif command == "unregister_telegram":
                    # Unregister a Telegram deployment
                    self.telegram_typing_manager.unregister_deployment(deployment_id)

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
    for deployment_id in list(manager.clients.keys()):
        await manager.stop_client(deployment_id)


web_app = FastAPI(lifespan=lifespan)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=1,
    allow_concurrent_inputs=100,
    timeout=3600,
)
@modal.asgi_app()
def gateway_app():
    return web_app
