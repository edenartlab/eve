import os
import logging

from fastapi import Request

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig

logger = logging.getLogger(__name__)


class TiktokClient(PlatformClient):
    TOOLS = [
        "tiktok_post",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate TikTok credentials and add TikTok tools"""
        try:
            # Validate that we have TikTok secrets
            if not secrets.tiktok or not secrets.tiktok.access_token:
                raise Exception("TikTok access token is required")

            # Add TikTok tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Invalid TikTok credentials: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for TikTok"""
        pass

    async def stop(self) -> None:
        """Stop TikTok client"""
        try:
            # Remove TikTok tools
            self.remove_tools()
        except Exception as e:
            print(f"Failed to remove TikTok tools: {e}")

    async def interact(self, request: Request) -> None:
        """Handle session interactions for TikTok"""
        # TikTok doesn't have real-time interactions like Discord
        pass

    async def handle_emission(self, emission) -> None:
        """Handle emissions for TikTok"""
        # TikTok doesn't have emissions like Discord
        pass