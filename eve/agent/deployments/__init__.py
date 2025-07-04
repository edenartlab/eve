from typing import TYPE_CHECKING, Optional
from fastapi import Request
from abc import ABC, abstractmethod

from eve.agent.agent import Agent
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig, Deployment


if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class PlatformClient(ABC):
    """Abstract base class for platform-specific client implementations"""

    # Class-level tool definitions
    TOOLS: dict[str, dict] = {}

    def __init__(
        self, agent: Optional[Agent] = None, deployment: Optional[Deployment] = None
    ):
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
        self.agent.get_collection().update_one(
            {"_id": self.agent.id}, 
            {"$set": {"tools.social_media_tools": True}, "$currentDate": {"updatedAt": True}}
        )
        self.agent.reload()
        
    def remove_tools(self) -> None:
        """Remove platform-specific tools from agent"""
        # self.agent.get_collection().update_one(
        #     {"_id": self.agent.id}, 
        #     {"$set": {"tools.social_media_tools": False}, "$currentDate": {"updatedAt": True}}
        # )
        ## Note: this is skipped for now because we don't want to remove the social media tools unless there are 0 deployments left
        pass

    @abstractmethod
    async def interact(self, request: Request) -> None:
        """Interact with the platform client"""
        pass

    @abstractmethod
    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the platform client"""
        pass
