from typing import TYPE_CHECKING
from fastapi import Request
from loguru import logger

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class TwitterClient(PlatformClient):
    TOOLS = [
        "tweet",
        "twitter_search",
        "twitter_mentions",
        "twitter_trends",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Twitter OAuth 2.0 credentials and add Twitter tools"""
        try:
            # Simple validation - just check we have the required OAuth 2.0 fields
            if not secrets.twitter.access_token:
                raise ValueError("Missing OAuth 2.0 access token")

            if not secrets.twitter.twitter_id:
                raise ValueError("Missing Twitter user ID")

            if not secrets.twitter.username:
                raise ValueError("Missing Twitter username")
            # Add Twitter tools to agent
            self.add_tools()

            return secrets, config
        except Exception as e:
            logger.error(f"Invalid Twitter credentials: {str(e)}")
            raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Twitter"""
        pass

    async def stop(self) -> None:
        """Stop Twitter client"""
        self.remove_tools()

    async def interact(self, request: Request) -> None:
        """Interact with the Twitter client"""
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Twitter client"""
        pass
