"""Discord V3 webhook-based deployment client."""

import json
import os

import aiohttp
from ably import AblyRest
from bson import ObjectId
from fastapi import Request
from loguru import logger

from eve.agent.deployments import PlatformClient
from eve.agent.deployments.discord_oauth import (
    create_agent_role,
    delete_agent_role,
    fetch_channel_name,
    fetch_guild_name,
    get_or_create_webhook,
)
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.models import (
    DeploymentConfig,
    DeploymentSecrets,
    DiscordChannelConfig,
    UpdateType,
)
from eve.api.errors import APIError
from eve.api.helpers import get_eden_creation_url
from eve.utils import prepare_result

db = os.getenv("DB", "STAGE").upper()


class DiscordV3Client(PlatformClient):
    """Discord V3 client using shared Eden bot with webhooks."""

    TOOLS = [
        "discord_post",
        "discord_search",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Set up webhooks and role for Discord V3 deployment."""
        if not config or not config.discord:
            raise APIError("Discord config is required", status_code=400)

        if not config.discord.guild_id:
            raise APIError("guild_id is required", status_code=400)

        if not config.discord.channel_configs:
            raise APIError("At least one channel is required", status_code=400)

        try:
            guild_id = config.discord.guild_id
            user_id = ObjectId(self.deployment.user) if self.deployment else None

            # Fetch guild name for display
            guild_name = await fetch_guild_name(guild_id)
            config.discord.guild_name = guild_name

            # Create role for agent mentions
            role_id = await create_agent_role(guild_id, self.agent.username)
            config.discord.role_id = role_id
            config.discord.role_name = self.agent.username

            # Create webhooks for each channel
            updated_channels = []
            for channel_cfg in config.discord.channel_configs:
                channel_id = channel_cfg.channel_id

                # Get or create webhook for this channel
                webhook_info = await get_or_create_webhook(
                    guild_id, channel_id, user_id
                )

                # Fetch channel name if not provided
                channel_name = channel_cfg.channel_name
                if not channel_name:
                    channel_name = await fetch_channel_name(channel_id)

                # Update channel config with webhook info
                updated_channel = DiscordChannelConfig(
                    channel_id=channel_id,
                    channel_name=channel_name,
                    access=channel_cfg.access,
                    webhook_id=webhook_info["id"],
                    webhook_token=webhook_info["token"],
                )
                updated_channels.append(updated_channel)

            config.discord.channel_configs = updated_channels

            # Add Discord tools to agent
            self.add_tools()

            logger.info(
                f"Discord V3 predeploy complete for guild {guild_id}, role {role_id}, {len(updated_channels)} channels"
            )

            return secrets, config

        except Exception as e:
            logger.error(f"Discord V3 predeploy failed: {e}", exc_info=True)
            raise APIError(
                f"Failed to set up Discord V3 deployment: {str(e)}", status_code=400
            )

    async def postdeploy(self) -> None:
        """Notify Discord gateway service via Ably."""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v3-{db}")

            await channel.publish(
                "command",
                {"command": "start", "deployment_id": str(self.deployment.id)},
            )
            logger.info(
                f"Notified Discord V3 gateway for deployment {self.deployment.id}"
            )
        except Exception as e:
            raise Exception(f"Failed to notify Discord V3 gateway: {e}")

    async def stop(self) -> None:
        """Stop Discord V3 client and clean up role."""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            # Notify gateway to stop
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v3-{db}")

            await channel.publish(
                "command",
                {"command": "stop", "deployment_id": str(self.deployment.id)},
            )

            # Clean up role if we have the info
            if (
                self.deployment.config
                and self.deployment.config.discord
                and self.deployment.config.discord.guild_id
                and self.deployment.config.discord.role_id
            ):
                await delete_agent_role(
                    self.deployment.config.discord.guild_id,
                    self.deployment.config.discord.role_id,
                )

            logger.info(f"Stopped Discord V3 deployment {self.deployment.id}")
        except Exception as e:
            logger.error(f"Error stopping Discord V3 deployment: {e}")
            raise Exception(f"Failed to stop Discord V3 client: {e}")

    async def interact(self, request: Request) -> None:
        """Handle session interactions for Discord V3"""
        try:
            from eve.api.api_requests import DeploymentInteractRequest

            # Parse the interaction request
            data = await request.json()
            interact_request = DeploymentInteractRequest(**data)

            # Forward the session request to the sessions API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{get_api_url()}/sessions/prompt",
                    json=interact_request.interaction.model_dump(),
                    headers={
                        "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                        "Content-Type": "application/json",
                        "X-Client-Platform": "discord_v3",
                        "X-Client-Deployment-Id": str(self.deployment.id),
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Failed to process session interaction: {error_text}"
                        )

                    logger.info(
                        f"Successfully handled Discord V3 session interaction for deployment {self.deployment.id}"
                    )

        except Exception as e:
            logger.error(
                f"Error handling Discord V3 interaction: {str(e)}", exc_info=True
            )
            raise

    async def handle_emission(self, emission) -> None:
        """Handle an emission from Discord V3 deployment (webhook-based)"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Extract channel context
            channel_id = emission.update_config.discord_channel_id

            if not channel_id:
                logger.error("Missing discord_channel_id in update_config")
                return

            # Build message payload
            payload = self._build_message_payload(emission)
            if not payload or not payload.get("content"):
                logger.debug("No content to send")
                return

            # Send via webhook
            from eve.agent.agent import Agent

            agent = Agent.from_mongo(self.deployment.agent)
            if not agent:
                raise Exception("Agent not found")

            await self._send_via_webhook(
                channel_id=channel_id,
                payload=payload,
                agent_name=agent.name,
                agent_image=agent.userImage,
            )

        except Exception as e:
            logger.error(f"Error handling Discord V3 emission: {str(e)}", exc_info=True)
            raise

    def _build_message_payload(self, emission) -> dict:
        """Build the message payload based on emission type"""
        payload = {}
        update_type = emission.type

        if update_type == UpdateType.ASSISTANT_MESSAGE:
            content = emission.content
            if content:
                payload["content"] = content

        elif update_type == UpdateType.TOOL_COMPLETE:
            result = emission.result
            if not result:
                logger.debug("No tool result to post")
                return None

            # Process result to extract media URLs
            processed_result = prepare_result(json.loads(result))

            if (
                processed_result.get("result")
                and len(processed_result["result"]) > 0
                and "output" in processed_result["result"][0]
            ):
                outputs = processed_result["result"][0]["output"]

                # Extract URLs from outputs
                urls = []
                for output in outputs[:4]:  # Discord supports up to 4 embeds
                    if isinstance(output, dict) and "url" in output:
                        urls.append(output["url"])

                if urls:
                    # Prepare message content with URLs
                    content = "\n".join(urls)
                    payload["content"] = content

                    # Get creation ID from the first output for Eden link
                    creation_id = None
                    if isinstance(outputs, list) and len(outputs) > 0:
                        creation_id = str(outputs[0].get("creation"))

                    # Add components for Eden link if creation_id exists
                    if creation_id:
                        eden_url = get_eden_creation_url(creation_id)
                        payload["components"] = [
                            {
                                "type": 1,  # Action Row
                                "components": [
                                    {
                                        "type": 2,  # Button
                                        "style": 5,  # Link
                                        "label": "View on Eden",
                                        "url": eden_url,
                                    }
                                ],
                            }
                        ]
                else:
                    logger.warning("No valid URLs found in tool result for Discord")
            else:
                logger.warning("Unexpected tool result structure for Discord emission")

        elif update_type == UpdateType.ERROR:
            error_msg = emission.error or "Unknown error occurred"
            payload["content"] = f"Error: {error_msg}"

        else:
            logger.debug(f"Ignoring emission type: {update_type}")
            return None

        return payload

    async def _send_via_webhook(
        self, channel_id: str, payload: dict, agent_name: str, agent_image: str
    ) -> None:
        """Send message via Discord webhook."""
        from eve.s3 import get_full_url

        # Find channel config with webhook
        channel_config = None
        if self.deployment.config and self.deployment.config.discord:
            for ch in self.deployment.config.discord.channel_configs or []:
                if ch.channel_id == channel_id:
                    channel_config = ch
                    break

        if not channel_config or not channel_config.webhook_id:
            raise Exception(f"No webhook configured for channel {channel_id}")

        webhook_token = channel_config.webhook_token

        # Get full URL for avatar (may be just a filename)
        avatar_url = (
            get_full_url(agent_image)
            if agent_image and not agent_image.startswith("http")
            else agent_image
        )

        # Build webhook payload
        webhook_payload = {
            "content": payload.get("content"),
            "username": agent_name,  # Custom name
            "avatar_url": avatar_url,  # Custom avatar
            "allowed_mentions": {"parse": ["users", "roles"]},
        }

        # Add components if present (buttons)
        if payload.get("components"):
            webhook_payload["components"] = payload["components"]

        # Post via webhook
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://discord.com/api/v10/webhooks/{channel_config.webhook_id}/{webhook_token}",
                json=webhook_payload,
            ) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    logger.error(f"Failed to send webhook message: {error_text}")
                    raise Exception(f"Failed to send webhook message: {error_text}")

                logger.info(
                    f"Successfully sent webhook message to channel {channel_id} as {agent_name}"
                )
