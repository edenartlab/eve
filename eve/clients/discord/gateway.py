from pathlib import Path
import modal
import asyncio
import json
import logging
import os
from typing import Dict, Optional
import websockets
import aiohttp
from ably import AblyRealtime
from eve import db
from eve.deploy import Deployment, ClientType
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


class DiscordGatewayClient:
    GATEWAY_VERSION = 10
    GATEWAY_URL = f"wss://gateway.discord.gg/?v={GATEWAY_VERSION}&encoding=json"

    def __init__(self, deployment: Deployment):
        if deployment.platform != ClientType.DISCORD_HTTP:
            raise ValueError("Deployment must be for Discord HTTP platform")

        self.deployment = deployment
        self.token = deployment.secrets.discord.token
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.heartbeat_interval: Optional[int] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect = True
        self._last_sequence = None
        self._session_id = None

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

    async def handle_message(self, data: dict):
        logger.info(
            f"Handling message for deployment {self.deployment.id} with data {data}"
        )
        logger.info(f"Deployment config: {self.deployment.config}")
        if not self.deployment.config:
            return

        # Skip messages from the bot itself
        print("AUTHOR ID", data.get("author", {}).get("id"))
        if (
            data.get("author", {}).get("id")
            == self.deployment.secrets.discord.application_id
        ):
            print("SKIPPING MESSAGE FROM BOT ITSELF")
            print("DEPLOYMENT ID", self.deployment.id)
            print("APPLICATION ID", self.deployment.secrets.discord.application_id)
            return

        channel_id = str(data["channel_id"])

        if self.deployment.config.discord.channel_allowlist:
            allowed_channels = [
                item.id for item in self.deployment.config.discord.channel_allowlist
            ]

            if channel_id not in allowed_channels:
                logger.info(
                    f"Message not in allowed channels, skipping for deployment {self.deployment.id}"
                )
                return

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
            "user_id": str(self.deployment.user),
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

    async def connect(self):
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

    async def setup_ably(self):
        """Set up Ably subscription for gateway commands"""

        async def message_handler(message):
            try:
                data = message.data
                if not isinstance(data, dict):
                    logger.warning(f"Received invalid message format: {data}")
                    return

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

                elif command == "reload":
                    # Reload a gateway client (stop and start)
                    if deployment_id in self.clients:
                        await self.stop_client(deployment_id)

                    deployment = Deployment.from_mongo(deployment_id)
                    if deployment:
                        await self.start_client(deployment)
                    else:
                        logger.error(f"Deployment not found: {deployment_id}")

            except Exception as e:
                logger.error(f"Error handling Ably message: {e}")

        # Subscribe to the channel
        await self.channel.subscribe(message_handler)
        logger.info("Subscribed to Ably channel for gateway commands")

    async def load_deployments(self):
        """Load all Discord HTTP deployments from database"""
        deployments = Deployment.find({"platform": ClientType.DISCORD_HTTP.value})
        for deployment in deployments:
            if deployment.secrets and deployment.secrets.discord.token:
                await self.start_client(deployment)

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
            client = self.clients.pop(deployment_id)
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
