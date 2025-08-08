from typing import TYPE_CHECKING
from fastapi import Request
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class PrintifyClient(PlatformClient):
    TOOLS = [
        "printify",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Printify credentials and add Printify tools"""
        # Validate that required fields are present
        if not secrets:
            raise ValueError("Deployment secrets are required")
            
        if not secrets.printify:
            raise ValueError("Printify secrets are required")
        
        if not secrets.printify.api_token:
            raise ValueError("api_token is required")
        
        if not secrets.printify.shop_id:
            raise ValueError("shop_id is required")

        print(f"Verified Printify credentials for shop ID: {secrets.printify.shop_id}")

        # Add Printify tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Printify"""
        pass

    async def stop(self) -> None:
        """Stop Printify client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        self.remove_tools()

    async def interact(self, request: Request) -> None:
        """Interact with the Printify client"""
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Printify client"""
        pass