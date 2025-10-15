from typing import TYPE_CHECKING
from fastapi import Request
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class ShopifyClient(PlatformClient):
    TOOLS = [
        "shopify",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Shopify credentials and add Shopify tools"""
        # For now, just validate that required fields are present
        if not secrets.shopify:
            raise ValueError("Shopify secrets are required")

        if not secrets.shopify.store_name:
            raise ValueError("store_name is required")

        if not secrets.shopify.access_token:
            raise ValueError("access_token is required")

        if not secrets.shopify.location_id:
            raise ValueError("location_id is required")

        # Add Shopify tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Shopify"""
        pass

    async def stop(self) -> None:
        """Stop Shopify client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        self.remove_tools()

    async def interact(self, request: Request) -> None:
        """Interact with the Shopify client"""
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Shopify client"""
        pass
