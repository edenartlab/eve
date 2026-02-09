from typing import TYPE_CHECKING

from fastapi import Request

from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentConfig, DeploymentSecrets

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


class MoltbookClient(PlatformClient):
    TOOLS = [
        "moltbook_post",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Moltbook credentials and add Moltbook tools"""
        if not secrets:
            raise ValueError("Deployment secrets are required")

        if not secrets.moltbook:
            raise ValueError("Moltbook secrets are required")

        if not secrets.moltbook.api_key:
            raise ValueError("api_key is required")

        # Validate the API key by calling the Moltbook API
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://www.moltbook.com/api/v1/agents/me",
                headers={"Authorization": f"Bearer {secrets.moltbook.api_key}"},
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Invalid Moltbook API key: {response.status_code} {response.text}"
                )

        # Add Moltbook tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Moltbook"""
        pass

    async def stop(self) -> None:
        """Stop Moltbook client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        self.remove_tools()

    async def interact(self, request: Request) -> None:
        """Interact with the Moltbook client"""
        raise NotImplementedError("Moltbook does not support interact")

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Moltbook client"""
        raise NotImplementedError("Moltbook does not support emissions")
