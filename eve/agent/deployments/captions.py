from typing import TYPE_CHECKING
from fastapi import Request
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class CaptionsClient(PlatformClient):
    TOOLS = [
        "captions",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Captions credentials and add Captions tools"""
        # Validate that required fields are present
        if not secrets:
            raise ValueError("Deployment secrets are required")
            
        if not secrets.captions:
            raise ValueError("Captions secrets are required")
        
        if not secrets.captions.api_key:
            raise ValueError("api_key is required")

        # Add Captions tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Captions"""
        pass

    async def stop(self) -> None:
        """Stop Captions client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        self.remove_tools()

    async def interact(self, request: Request) -> None:
        """Interact with the Captions client"""
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Captions client"""
        pass