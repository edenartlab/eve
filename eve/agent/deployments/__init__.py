from enum import Enum
from typing import Optional
from pydantic import BaseModel
from bson import ObjectId
from abc import ABC, abstractmethod

from eve.agent.agent import Agent
from eve.agent.deployments.discord import (
    DiscordClient,
    DeploymentSecretsDiscord,
    DeploymentSettingsDiscord,
)
from eve.agent.deployments.farcaster import (
    FarcasterClient,
    DeploymentSecretsFarcaster,
    DeploymentSettingsFarcaster,
)
from eve.agent.deployments.telegram import (
    TelegramClient,
    DeploymentSecretsTelegram,
    DeploymentSettingsTelegram,
)
from eve.agent.deployments.twitter import (
    TwitterClient,
    DeploymentSecretsTwitter,
    DeploymentSettingsTwitter,
)
from eve.mongo import Collection, Document


class ClientType(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
    TWITTER = "twitter"


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
