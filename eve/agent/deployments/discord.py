import os
from typing import List, Optional

import aiohttp
from ably import AblyRest
from pydantic import BaseModel

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient, DeploymentSecrets, DeploymentConfig

db = os.getenv("DB", "STAGE").upper()


class DiscordAllowlistItem(BaseModel):
    id: str
    note: Optional[str] = None


class DeploymentSettingsDiscord(BaseModel):
    oauth_client_id: Optional[str] = None
    oauth_url: Optional[str] = None
    channel_allowlist: Optional[List[DiscordAllowlistItem]] = None
    read_access_channels: Optional[List[DiscordAllowlistItem]] = None


class DeploymentSecretsDiscord(BaseModel):
    token: str
    application_id: Optional[str] = None


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
