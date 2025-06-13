from enum import Enum
import os
from typing import List, Optional
import aiohttp
from ably import AblyRest
from abc import ABC, abstractmethod

from bson import ObjectId
from pydantic import BaseModel

from eve.agent.agent import Agent
from eve.mongo import Collection, Document
from eve.api.errors import APIError

db = os.getenv("DB", "STAGE").upper()


class ClientType(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
    TWITTER = "twitter"


class AllowlistItem(BaseModel):
    id: str
    note: Optional[str] = None


class DeploymentSettingsDiscord(BaseModel):
    oauth_client_id: Optional[str] = None
    oauth_url: Optional[str] = None
    channel_allowlist: Optional[List[AllowlistItem]] = None
    read_access_channels: Optional[List[AllowlistItem]] = None


class DeploymentSettingsTelegram(BaseModel):
    topic_allowlist: Optional[List[AllowlistItem]] = None


class DeploymentSettingsFarcaster(BaseModel):
    webhook_id: Optional[str] = None
    auto_reply: Optional[bool] = False


class DeploymentSettingsTwitter(BaseModel):
    username: Optional[str] = None


class DeploymentSecretsDiscord(BaseModel):
    token: str
    application_id: Optional[str] = None


class DeploymentSecretsTelegram(BaseModel):
    token: str
    webhook_secret: Optional[str] = None


class DeploymentSecretsFarcaster(BaseModel):
    mnemonic: str
    neynar_webhook_secret: Optional[str] = None


class DeploymentSecretsTwitter(BaseModel):
    user_id: str
    bearer_token: str
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str


class DeploymentSecrets(BaseModel):
    discord: DeploymentSecretsDiscord | None = None
    telegram: DeploymentSecretsTelegram | None = None
    farcaster: DeploymentSecretsFarcaster | None = None
    twitter: DeploymentSecretsTwitter | None = None


class DeploymentConfig(BaseModel):
    discord: DeploymentSettingsDiscord | None = None
    telegram: DeploymentSettingsTelegram | None = None
    farcaster: DeploymentSettingsFarcaster | None = None
    twitter: DeploymentSettingsTwitter | None = None


@Collection("deployments2")
class Deployment(Document):
    agent: ObjectId
    user: ObjectId
    platform: ClientType
    valid: Optional[bool] = None
    secrets: Optional[DeploymentSecrets]
    config: Optional[DeploymentConfig]

    def __init__(self, **data):
        # Convert string to ClientType enum if needed
        if "platform" in data and isinstance(data["platform"], str):
            data["platform"] = ClientType(data["platform"])
        super().__init__(**data)

    def model_dump(self, *args, **kwargs):
        """Override model_dump to convert enum to string for MongoDB"""
        data = super().model_dump(*args, **kwargs)
        if "platform" in data and isinstance(data["platform"], ClientType):
            data["platform"] = data["platform"].value
        return data

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent", 1), ("platform", 1)], unique=True)


class PlatformClient(ABC):
    """Abstract base class for platform-specific client implementations"""

    # Class-level tool definitions
    TOOLS: dict[str, dict] = {}

    def __init__(self, agent: Agent, deployment: Optional[Deployment] = None):
        self.agent = agent
        self.deployment = deployment

    @abstractmethod
    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Platform-specific validation and setup before deployment"""
        pass

    @abstractmethod
    async def postdeploy(self) -> None:
        """Platform-specific actions after deployment"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the platform client"""
        pass

    def add_tools(self) -> None:
        """Add platform-specific tools to agent"""
        if not self.agent.tools:
            self.agent.tools = {}
            self.agent.add_base_tools = True

        for tool_name, tool_config in self.TOOLS.items():
            self.agent.tools[tool_name] = {
                "parameters": {
                    "agent": {"default": str(self.agent.id), "hide_from_agent": True}
                }
            }
        self.agent.save()

    def remove_tools(self) -> None:
        """Remove platform-specific tools from agent"""
        if self.agent.tools:
            for tool_name in self.TOOLS.keys():
                self.agent.tools.pop(tool_name, None)
            self.agent.save()


class DiscordClient(PlatformClient):
    TOOLS = {
        "discord_search": {},
        "discord_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Discord token and setup OAuth"""
        headers = {"Authorization": f"Bot {secrets.discord.token}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://discord.com/api/v10/users/@me", headers=headers
            ) as response:
                if response.status != 200:
                    raise APIError("Invalid Discord token", status_code=400)

                # Get application ID if not provided
                if not secrets.discord.application_id:
                    bot_data = await response.json()
                    application_id = bot_data.get("id")
                    if application_id:
                        secrets.discord.application_id = application_id

                        # Setup Discord config
                        if not config:
                            config = DeploymentConfig()
                        if not config.discord:
                            config.discord = DeploymentSettingsDiscord()

                        # Create OAuth URL with the same permissions integer
                        permissions_integer = "309237771264"
                        oauth_url = f"https://discord.com/oauth2/authorize?client_id={application_id}&permissions={permissions_integer}&integration_type=0&scope=bot"

                        config.discord.oauth_client_id = application_id
                        config.discord.oauth_url = oauth_url

        # Add Discord tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """Notify Discord gateway service via Ably"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {"command": "start", "deployment_id": str(self.deployment.id)},
            )
            print(f"Sent start command for deployment {self.deployment.id} via Ably")
        except Exception as e:
            raise Exception(f"Failed to notify gateway service: {e}")

    async def stop(self) -> None:
        """Stop Discord client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {"command": "stop", "deployment_id": str(self.deployment.id)},
            )
            print(f"Sent stop command for deployment {self.deployment.id} via Ably")

            # Remove Discord tools
            self.remove_tools()

        except Exception as e:
            print(f"Failed to notify gateway service: {e}")


class TelegramClient(PlatformClient):
    TOOLS = {
        "telegram_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Telegram token, generate webhook secret and add Telegram tools"""
        from telegram import Bot
        import secrets as python_secrets

        # Validate bot token
        try:
            bot = Bot(secrets.telegram.token)
            bot_info = await bot.get_me()
            print(f"Verified Telegram bot: {bot_info.username}")
        except Exception as e:
            raise APIError(f"Invalid Telegram token: {str(e)}", status_code=400)

        webhook_secret = python_secrets.token_urlsafe(32)
        secrets.telegram.webhook_secret = webhook_secret

        # Add Telegram tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """Set Telegram webhook and notify gateway"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        from telegram import Bot

        webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/telegram"

        # Update bot webhook
        response = await Bot(self.deployment.secrets.telegram.token).set_webhook(
            url=webhook_url,
            secret_token=self.deployment.secrets.telegram.webhook_secret,
            drop_pending_updates=True,
            max_connections=100,
        )

        if not response:
            raise Exception("Failed to set Telegram webhook")

        # Notify gateway about the new deployment
        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {
                    "command": "register_telegram",
                    "deployment_id": str(self.deployment.id),
                    "token": self.deployment.secrets.telegram.token,
                },
            )
            print(
                f"Sent Telegram registration command for deployment {self.deployment.id} via Ably"
            )
        except Exception as e:
            raise Exception(f"Failed to notify gateway service for Telegram: {e}")

    async def stop(self) -> None:
        """Stop Telegram client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {
                    "command": "unregister_telegram",
                    "deployment_id": str(self.deployment.id),
                },
            )
            print(
                f"Sent unregister command for Telegram deployment {self.deployment.id} via Ably"
            )

            self.remove_tools()
        except Exception as e:
            print(f"Failed to notify gateway service for Telegram unregistration: {e}")


class FarcasterClient(PlatformClient):
    TOOLS = {}  # No tools for Farcaster yet

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Verify Farcaster credentials"""
        try:
            from farcaster import Warpcast

            client = Warpcast(mnemonic=secrets.farcaster.mnemonic)

            # Test the credentials by getting user info
            user_info = client.get_me()
            print(f"Verified Farcaster credentials for user: {user_info}")
        except Exception as e:
            raise APIError(f"Invalid Farcaster credentials: {str(e)}", status_code=400)

        # Generate webhook secret if not provided
        if not secrets.farcaster.neynar_webhook_secret:
            import secrets as python_secrets

            webhook_secret = python_secrets.token_urlsafe(32)
            secrets.farcaster.neynar_webhook_secret = webhook_secret

        return secrets, config

    async def postdeploy(self) -> None:
        """Register webhook with Neynar"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/farcaster"

        async with aiohttp.ClientSession() as session:
            # Get Neynar API key from environment
            neynar_api_key = os.getenv("NEYNAR_API_KEY")
            if not neynar_api_key:
                raise Exception("NEYNAR_API_KEY not found in environment")

            headers = {
                "x-api-key": f"{neynar_api_key}",
                "Content-Type": "application/json",
            }

            # Get user info for webhook registration
            from farcaster import Warpcast

            client = Warpcast(mnemonic=self.deployment.secrets.farcaster.mnemonic)
            user_info = client.get_me()

            webhook_data = {
                "name": f"eden-{self.deployment.id}",
                "url": webhook_url,
                "subscription": {"cast.created": {"mentioned_fids": [user_info.fid]}},
            }

            async with session.post(
                "https://api.neynar.com/v2/farcaster/webhook",
                headers=headers,
                json=webhook_data,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to register Neynar webhook: {error_text}")

                webhook_response = await response.json()
                webhook_id = webhook_response.get("webhook", {}).get("webhook_id")
                webhook_secret = (
                    webhook_response.get("webhook", {})
                    .get("secrets", [{}])[0]
                    .get("value")
                )

                if not webhook_id:
                    raise Exception("No webhook_id in response")

                # Update the webhook secret in deployment secrets
                self.deployment.secrets.farcaster.neynar_webhook_secret = webhook_secret

                # Store webhook ID in deployment for later cleanup
                if not self.deployment.config:
                    self.deployment.config = DeploymentConfig()
                if not self.deployment.config.farcaster:
                    self.deployment.config.farcaster = DeploymentSettingsFarcaster()

                self.deployment.config.farcaster.webhook_id = webhook_id
                self.deployment.save()

                print(
                    f"Registered Neynar webhook {webhook_id} for deployment {self.deployment.id}"
                )

    async def stop(self) -> None:
        """Stop Farcaster client by unregistering webhook from Neynar"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        if self.deployment.config and self.deployment.config.farcaster:
            webhook_id = getattr(self.deployment.config.farcaster, "webhook_id", None)
            if webhook_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        neynar_api_key = os.getenv("NEYNAR_API_KEY")
                        headers = {
                            "x-api-key": f"{neynar_api_key}",
                            "Content-Type": "application/json",
                        }

                        webhook_data = {"webhook_id": webhook_id}

                        async with session.delete(
                            "https://api.neynar.com/v2/farcaster/webhook",
                            headers=headers,
                            json=webhook_data,
                        ) as response:
                            if response.status == 200:
                                print(
                                    f"Successfully unregistered Neynar webhook {webhook_id}"
                                )
                                # Clear the webhook_id from config after successful deletion
                                self.deployment.config.farcaster.webhook_id = None
                                self.deployment.save()
                            else:
                                error_text = await response.text()
                                print(f"Failed to unregister webhook: {error_text}")

                except Exception as e:
                    print(f"Error unregistering Neynar webhook: {e}")


class TwitterClient(PlatformClient):
    TOOLS = {
        "tweet": {},
        "twitter_mentions": {},
        "twitter_search": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Twitter credentials and add Twitter tools to agent"""
        import tweepy

        # Validate Twitter credentials
        try:
            # Create Twitter client with OAuth 1.0a
            auth = tweepy.OAuth1UserHandler(
                consumer_key=secrets.twitter.consumer_key,
                consumer_secret=secrets.twitter.consumer_secret,
                access_token=secrets.twitter.access_token,
                access_token_secret=secrets.twitter.access_token_secret,
            )
            api = tweepy.API(auth)

            # Test the credentials by getting user info
            user = api.verify_credentials()
            print(f"Verified Twitter credentials for user: @{user.screen_name}")

            # Also test with v2 API using bearer token
            client = tweepy.Client(
                bearer_token=secrets.twitter.bearer_token,
                consumer_key=secrets.twitter.consumer_key,
                consumer_secret=secrets.twitter.consumer_secret,
                access_token=secrets.twitter.access_token,
                access_token_secret=secrets.twitter.access_token_secret,
            )

            # Test v2 API
            me = client.get_me()
            print(f"Verified Twitter v2 API for user: @{me.data.username}")

        except Exception as e:
            raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

        # Add Twitter tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Twitter"""
        pass

    async def stop(self) -> None:
        """Stop Twitter client by removing tools"""
        self.remove_tools()


def get_platform_client(
    agent: Agent, platform: ClientType, deployment: Optional[Deployment] = None
) -> PlatformClient:
    """Helper function to get the appropriate platform client"""
    if platform == ClientType.DISCORD:
        return DiscordClient(agent=agent, deployment=deployment)
    elif platform == ClientType.TELEGRAM:
        return TelegramClient(agent=agent, deployment=deployment)
    elif platform == ClientType.FARCASTER:
        return FarcasterClient(agent=agent, deployment=deployment)
    elif platform == ClientType.TWITTER:
        return TwitterClient(agent=agent, deployment=deployment)


async def predeploy_platform(
    agent: Agent,
    secrets: DeploymentSecrets,
    config: DeploymentConfig,
    platform: ClientType,
):
    """Platform-specific validation, secret modification, and agent tool setup"""
    # Create deployment object for client
    deployment = Deployment(
        agent=agent.id,
        user=agent.user,
        platform=platform,
        secrets=secrets,
        config=config,
    )

    # Get appropriate platform client
    client = get_platform_client(agent=agent, platform=platform, deployment=deployment)

    # Run predeploy
    return await client.predeploy(secrets=secrets, config=config)


async def postdeploy_platform(
    agent: Agent,
    deployment: Deployment,
    platform: ClientType,
):
    """Platform-specific actions that require the deployment object"""
    # Get appropriate platform client
    client = get_platform_client(agent=agent, platform=platform, deployment=deployment)

    # Run postdeploy
    await client.postdeploy()


async def stop_client(agent: Agent, deployment: Deployment, platform: ClientType):
    """Stop a Modal client. For Discord HTTP, notify the gateway service via Ably."""

    # Get appropriate platform client
    client = get_platform_client(agent=agent, platform=platform, deployment=deployment)
    # Run stop
    await client.stop()
