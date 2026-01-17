"""Discord V3 webhook-based deployment client."""

import base64
import json
import os
from typing import Dict, List, Optional

import aiohttp
from ably import AblyRest
from bson import ObjectId
from fastapi import Request
from loguru import logger

from eve.agent.deployments import PlatformClient
from eve.agent.deployments.discord_gateway import (
    get_or_create_discord_message,
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
from eve.user import DiscordGuild, GuildWebhook
from eve.utils import prepare_result

db = os.getenv("DB", "STAGE").upper()


def encrypt_state(data: Dict) -> str:
    """Encode OAuth state data (just base64, not actual encryption)."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decrypt_state(encoded: str) -> Dict:
    """Decode OAuth state data."""
    return json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())


# Permission constants
ADMINISTRATOR = 0x8
MANAGE_GUILD = 0x20


def user_can_manage_guild_permissions(permissions_string: str) -> bool:
    """
    Check if user has permission to configure a guild based on permissions bitfield.

    Args:
        permissions_string: Permission bitfield from Discord OAuth

    Returns:
        True if user has ADMINISTRATOR or MANAGE_GUILD permissions
    """
    try:
        permissions = int(permissions_string)
    except (ValueError, TypeError):
        return False

    # Check for ADMINISTRATOR or MANAGE_GUILD
    return bool((permissions & ADMINISTRATOR) or (permissions & MANAGE_GUILD))


async def fetch_discord_user(access_token: str) -> Dict:
    """Fetch Discord user info using access token."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch user: {await response.text()}")
            return await response.json()


async def fetch_user_guilds(access_token: str) -> List[DiscordGuild]:
    """Fetch user's guilds using access token."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://discord.com/api/v10/users/@me/guilds",
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch guilds: {await response.text()}")

            guilds_data = await response.json()

            # Convert to DiscordGuild objects
            guilds = []
            for g in guilds_data:
                guild = DiscordGuild(
                    id=g["id"],
                    name=g["name"],
                    permissions=str(g["permissions"]),
                    owner=g.get("owner", False),
                    icon=g.get("icon"),
                    has_eden_bot=False,  # Will be checked separately
                )
                guilds.append(guild)

            return guilds


async def check_bot_in_guild(guild_id: str) -> bool:
    """Check if Eden bot is in this guild."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        return False

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://discord.com/api/v10/guilds/{guild_id}",
            headers={"Authorization": f"Bot {bot_token}"},
        ) as response:
            if response.status == 200:
                return True
            if response.status in [403, 404]:
                return False

            logger.warning(
                f"Unknown status checking bot in guild {guild_id}: {response.status}"
            )
            return False


async def create_agent_role(guild_id: str, agent_name: str) -> str:
    """Create a Discord role for this agent, or return existing role ID."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    async with aiohttp.ClientSession() as session:
        # Check if role already exists
        async with session.get(
            f"https://discord.com/api/v10/guilds/{guild_id}/roles",
            headers={"Authorization": f"Bot {bot_token}"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch roles: {await response.text()}")

            roles = await response.json()

            # Look for existing role with this name
            existing_role = next((r for r in roles if r["name"] == agent_name), None)

            if existing_role:
                logger.info(
                    f"Role {agent_name} already exists in guild {guild_id}: {existing_role['id']}"
                )
                return existing_role["id"]

        # Create new role
        async with session.post(
            f"https://discord.com/api/v10/guilds/{guild_id}/roles",
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            },
            json={
                "name": agent_name,
                "permissions": "0",  # No permissions
                "color": 0x5865F2,  # Discord blue
                "hoist": False,  # Don't display separately
                "mentionable": True,  # Allow @mentions
            },
        ) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Failed to create role: {await response.text()}")

            role = await response.json()
            logger.info(f"Created role {agent_name} in guild {guild_id}: {role['id']}")
            return role["id"]


async def get_or_create_webhook(
    guild_id: str, channel_id: str, user_id: ObjectId
) -> Dict[str, str]:
    """Get or create webhook for this channel."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    application_id = os.getenv("DISCORD_OAUTH_CLIENT_ID")

    # Check if we have it cached
    cached = GuildWebhook.get_for_channel(guild_id, channel_id)

    if cached:
        # Verify it still exists
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://discord.com/api/v10/webhooks/{cached.webhook_id}",
                    headers={"Authorization": f"Bot {bot_token}"},
                ) as response:
                    if response.status == 200:
                        logger.info(
                            f"Using cached webhook for channel {channel_id}: {cached.webhook_id}"
                        )
                        return {
                            "id": cached.webhook_id,
                            "token": cached.webhook_token,
                            "url": cached.webhook_url,
                        }
        except Exception as e:
            logger.warning(f"Cached webhook invalid, will recreate: {e}")
            # Webhook was deleted, clean up
            cached.delete()

    # Check Discord for an existing managed webhook
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://discord.com/api/v10/channels/{channel_id}/webhooks",
                headers={"Authorization": f"Bot {bot_token}"},
            ) as response:
                if response.status == 200:
                    webhooks = await response.json()
                    managed_webhook = None
                    for webhook in webhooks:
                        if not webhook.get("token"):
                            continue
                        if (
                            application_id
                            and webhook.get("application_id") == application_id
                        ):
                            managed_webhook = webhook
                            break
                    if not managed_webhook:
                        for webhook in webhooks:
                            if (
                                webhook.get("token")
                                and webhook.get("name") == "Eden Agent Webhook"
                            ):
                                managed_webhook = webhook
                                break

                    if managed_webhook:
                        webhook_id = managed_webhook["id"]
                        webhook_token = managed_webhook["token"]
                        webhook_url = managed_webhook.get("url") or (
                            f"https://discord.com/api/v10/webhooks/{webhook_id}/{webhook_token}"
                        )
                        webhook_doc = GuildWebhook(
                            guild_id=guild_id,
                            channel_id=channel_id,
                            webhook_id=webhook_id,
                            webhook_token=webhook_token,
                            webhook_url=webhook_url,
                            created_by=user_id,
                        )
                        webhook_doc.save(
                            upsert_filter={
                                "guild_id": guild_id,
                                "channel_id": channel_id,
                            }
                        )
                        logger.info(
                            f"Reusing existing webhook for channel {channel_id}: {webhook_id}"
                        )
                        return {
                            "id": webhook_id,
                            "token": webhook_token,
                            "url": webhook_url,
                        }
                else:
                    logger.warning(
                        f"Failed to list webhooks for channel {channel_id}: {await response.text()}"
                    )
    except Exception as e:
        logger.warning(f"Failed to inspect existing webhooks: {e}")

    # Create new webhook
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://discord.com/api/v10/channels/{channel_id}/webhooks",
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            },
            json={
                "name": "Eden Agent Webhook",
            },
        ) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Failed to create webhook: {await response.text()}")

            webhook = await response.json()

            # Cache it
            webhook_doc = GuildWebhook(
                guild_id=guild_id,
                channel_id=channel_id,
                webhook_id=webhook["id"],
                webhook_token=webhook["token"],
                webhook_url=webhook["url"],
                created_by=user_id,
            )
            webhook_doc.save(
                upsert_filter={"guild_id": guild_id, "channel_id": channel_id}
            )

            logger.info(f"Created webhook for channel {channel_id}: {webhook['id']}")
            return {
                "id": webhook["id"],
                "token": webhook["token"],
                "url": webhook["url"],
            }


async def fetch_channel_name(channel_id: str) -> str:
    """Fetch channel name from Discord API."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://discord.com/api/v10/channels/{channel_id}",
            headers={"Authorization": f"Bot {bot_token}"},
        ) as response:
            if response.status == 200:
                channel = await response.json()
                return channel.get("name", "unknown")
            return "unknown"


async def fetch_guild_name(guild_id: str) -> str:
    """Fetch guild name from Discord API."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://discord.com/api/v10/guilds/{guild_id}",
            headers={"Authorization": f"Bot {bot_token}"},
        ) as response:
            if response.status == 200:
                guild = await response.json()
                return guild.get("name", "unknown")
            return "unknown"


async def fetch_guild_channels(guild_id: str) -> List[Dict]:
    """Fetch channels in a guild."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://discord.com/api/v10/guilds/{guild_id}/channels",
            headers={"Authorization": f"Bot {bot_token}"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch channels: {await response.text()}")

            channels = await response.json()

            # Filter to text channels only (type 0 = GUILD_TEXT, type 5 = GUILD_NEWS)
            text_channels = [ch for ch in channels if ch["type"] in [0, 5]]

            return text_channels


async def exchange_oauth_code(code: str, redirect_uri: str) -> Dict:
    """Exchange OAuth code for tokens."""
    client_id = os.getenv("DISCORD_OAUTH_CLIENT_ID")
    client_secret = os.getenv("DISCORD_OAUTH_CLIENT_SECRET")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://discord.com/api/oauth2/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to exchange code: {await response.text()}")

            return await response.json()


async def delete_agent_role(guild_id: str, role_id: str) -> bool:
    """Delete an agent role from a guild."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"https://discord.com/api/v10/guilds/{guild_id}/roles/{role_id}",
                headers={"Authorization": f"Bot {bot_token}"},
            ) as response:
                if response.status == 204:
                    logger.info(f"Deleted role {role_id} from guild {guild_id}")
                    return True
                logger.warning(
                    f"Failed to delete role {role_id}: {await response.text()}"
                )
                return False
    except Exception as e:
        logger.error(f"Error deleting role: {e}")
        return False


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
            logger.error("Discord V3 predeploy failed: {}", str(e), exc_info=True)
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

            response_data = await self._send_via_webhook(
                channel_id=channel_id,
                payload=payload,
                agent_name=agent.name,
                agent_image=agent.userImage,
            )

            # Record webhook message attribution for later ingestion
            if response_data and response_data.get("id"):
                try:
                    message_data = dict(response_data)
                    if self.deployment and self.deployment.config:
                        guild_id = (
                            self.deployment.config.discord.guild_id
                            if self.deployment.config.discord
                            else None
                        )
                        if guild_id and not message_data.get("guild_id"):
                            message_data["guild_id"] = guild_id
                    if not message_data.get("channel_id"):
                        message_data["channel_id"] = channel_id

                    discord_msg, _ = get_or_create_discord_message(message_data)
                    if discord_msg:
                        discord_msg.update(
                            source_agent_id=str(agent.id),
                            source_deployment_id=str(self.deployment.id)
                            if self.deployment
                            else None,
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to record webhook attribution for message: {e}"
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
    ) -> Optional[dict]:
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
                params={"wait": "true"},
                json=webhook_payload,
            ) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    logger.error(f"Failed to send webhook message: {error_text}")
                    raise Exception(f"Failed to send webhook message: {error_text}")

                response_data = None
                if response.status == 200:
                    try:
                        response_data = await response.json()
                    except Exception as e:
                        logger.warning(f"Failed to parse webhook response JSON: {e}")

                logger.info(
                    f"Successfully sent webhook message to channel {channel_id} as {agent_name}"
                )
                return response_data
