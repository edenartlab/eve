"""Discord OAuth and webhook management utilities."""

import base64
import json
import os
from typing import Dict, List

import aiohttp
from bson import ObjectId
from loguru import logger

from eve.user import DiscordGuild, GuildWebhook


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
            elif response.status in [403, 404]:
                return False
            else:
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
    # Check if we have it cached
    cached = GuildWebhook.get_for_channel(guild_id, channel_id)

    if cached:
        # Verify it still exists
        try:
            bot_token = os.getenv("DISCORD_BOT_TOKEN")
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

    # Create new webhook
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
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
            webhook_doc.save()

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
            else:
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
            else:
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
                else:
                    logger.warning(
                        f"Failed to delete role {role_id}: {await response.text()}"
                    )
                    return False
    except Exception as e:
        logger.error(f"Error deleting role: {e}")
        return False
