import os

import aiohttp
from ably import AblyRest

from eve.api.errors import APIError
from eve.agent.deployments import (
    PlatformClient,
    DeploymentSecrets,
    DeploymentConfig,
)

db = os.getenv("DB", "STAGE").upper()


class DiscordClient(PlatformClient):
    TOOLS = {
        "discord_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Discord token and add Discord tools"""
        try:
            # Validate bot token
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {secrets.discord.token}"},
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Invalid token: {await response.text()}")
                    bot_info = await response.json()
                    print(f"Verified Discord bot: {bot_info['username']}")

            # Add Discord tools to agent
            self.add_tools()

            return secrets, config
        except Exception as e:
            raise APIError(f"Invalid Discord token: {str(e)}", status_code=400)

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
