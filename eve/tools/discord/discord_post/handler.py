import io
import os
from typing import List, Optional

import aiohttp
import discord
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.deployments.discord_gateway import convert_usernames_to_discord_mentions
from eve.agent.session.models import Deployment, Session
from eve.tool import ToolContext

DISCORD_MAX_LENGTH = 2000


def split_content_into_chunks(
    content: str, max_length: int = DISCORD_MAX_LENGTH
) -> List[str]:
    """
    Split content into chunks of max_length, preferring to break at spaces.

    Args:
        content: The text content to split
        max_length: Maximum length per chunk (default: Discord's 2000 char limit)

    Returns:
        List of content chunks, each <= max_length
    """
    if not content or len(content) <= max_length:
        return [content] if content else []

    chunks = []
    remaining = content

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find a good break point (prefer space, then newline)
        chunk = remaining[:max_length]

        # Look for the last space or newline within the chunk
        break_point = -1
        for delimiter in ["\n", " "]:
            last_pos = chunk.rfind(delimiter)
            if last_pos > max_length // 2:  # Only use if in the second half
                break_point = last_pos
                break

        if break_point > 0:
            # Split at the delimiter
            chunks.append(remaining[:break_point])
            remaining = remaining[break_point + 1 :]  # Skip the delimiter
        else:
            # No good break point, hard split at max_length
            chunks.append(chunk)
            remaining = remaining[max_length:]

    return chunks


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)

    # Try to find Discord V3 deployment first, then fallback to legacy Discord
    deployment = Deployment.load(agent=agent.id, platform="discord_v3")
    if not deployment:
        deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")

    # Get parameters
    channel_id = context.args.get("channel_id")
    discord_user_id = context.args.get("discord_user_id")
    content = context.args["content"]
    media_urls = context.args.get("media_urls", [])
    reply_to = context.args.get("reply_to")

    # Validate parameters
    if not discord_user_id and not channel_id:
        raise Exception("Either channel_id or discord_user_id must be provided")

    # Content can be empty only if media URLs are provided
    if (not content or not content.strip()) and not media_urls:
        raise Exception("Either content or media_urls must be provided")

    # Convert @username mentions to Discord <@id> format before sending
    if content:
        content = convert_usernames_to_discord_mentions(content)

    # Check if this is a V3 webhook-based deployment
    is_webhook_deployment = (
        deployment.config
        and deployment.config.discord
        and deployment.config.discord.guild_id is not None
    )

    # Check if we can skip allowlist validation (fast path)
    # If channel_id matches session's discord_channel_id, it was already validated
    skip_allowlist_check = False
    if channel_id and context.session:
        try:
            session = Session.from_mongo(context.session)
            if session and session.discord_channel_id == channel_id:
                skip_allowlist_check = True
                logger.info(
                    f"discord_post: Skipping allowlist check - channel {channel_id} matches session's discord_channel_id"
                )
        except Exception as e:
            logger.warning(
                f"discord_post: Failed to load session for allowlist check: {e}"
            )

    # V3 Webhook-based posting (no bot login required)
    if is_webhook_deployment and channel_id and not discord_user_id:
        logger.info("discord_post: Using webhook for Discord V3 deployment")
        return await send_webhook_message(
            deployment=deployment,
            agent=agent,
            channel_id=channel_id,
            content=content,
            media_urls=media_urls,
            skip_allowlist_check=skip_allowlist_check,
        )

    # Legacy token-based posting (requires bot login)
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Download media files if provided
        files = []
        if media_urls:
            files = await download_media_files(media_urls)
            # If download failed, fallback to appending URLs to content
            if not files:
                media_content = "\n".join(media_urls)
                content = f"{content}\n\n{media_content}"

        if discord_user_id:
            # Send DM to user
            return await send_dm(client, discord_user_id, content, files)
        else:
            # Send message to channel (existing functionality)
            return await send_channel_message(
                client,
                deployment,
                channel_id,
                content,
                files,
                reply_to,
                skip_allowlist_check=skip_allowlist_check,
            )

    finally:
        await client.close()


async def download_media_files(media_urls: list) -> list:
    """Download media files from URLs and return Discord file objects."""
    files = []

    async with aiohttp.ClientSession() as session:
        for url in media_urls:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Get filename from URL or use a default
                        filename = os.path.basename(url.split("?")[0])
                        if not filename or "." not in filename:
                            # Determine extension from content-type
                            content_type = response.headers.get("content-type", "")
                            if "image" in content_type:
                                ext = content_type.split("/")[-1]
                                filename = f"image.{ext}"
                            elif "video" in content_type:
                                ext = content_type.split("/")[-1]
                                filename = f"video.{ext}"
                            else:
                                filename = "attachment"

                        # Read file content
                        file_data = await response.read()
                        file_obj = discord.File(
                            io.BytesIO(file_data), filename=filename
                        )
                        files.append(file_obj)
            except Exception as e:
                # If any file fails to download, return empty list to trigger fallback
                logger.error(f"Failed to download media from {url}: {str(e)}")
                return []

    return files


async def send_dm(
    client: discord.Client, discord_user_id: str, content: str, files: list = None
):
    """Send a direct message to a Discord user."""
    try:
        # Validate that discord_user_id is a numeric ID
        try:
            user_id_int = int(discord_user_id)
        except ValueError:
            raise Exception(
                f"discord_user_id must be a numeric Discord ID (e.g., '987654321'), not a username. Got: '{discord_user_id}'"
            )

        # Get the user object
        user = await client.fetch_user(user_id_int)

        # Split content into chunks (Discord limit is 2000 chars)
        chunks = split_content_into_chunks(content)
        if not chunks:
            chunks = [""]  # Allow empty content if files are provided

        messages = []
        last_idx = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            # Attach files to the last message so they appear at the end
            if i == last_idx and files:
                message = await user.send(content=chunk, files=files)
            else:
                message = await user.send(content=chunk)
            messages.append(message)

        # Return URLs for all messages
        return {
            "output": [
                {
                    "url": f"https://discord.com/channels/@me/{user.dm_channel.id}/{msg.id}",
                }
                for msg in messages[
                    :1
                ]  # return just the first url if messages are split
            ]
        }

    except discord.Forbidden:
        # User has DMs disabled or bot is blocked
        raise Exception(
            f"Cannot send DM to user {discord_user_id}: DMs disabled or bot blocked"
        )
    except discord.NotFound:
        raise Exception(f"User {discord_user_id} not found")
    except Exception as e:
        raise Exception(f"Failed to send DM to user {discord_user_id}: {str(e)}")


async def check_thread_parent_allowlist(
    channel_id: str, allowed_ids: List[str], token: str
) -> tuple[bool, Optional[str]]:
    """
    Check if a channel is a thread with an allowlisted parent.

    Args:
        channel_id: The Discord channel ID to check
        allowed_ids: List of allowed channel IDs
        token: Discord bot token for API calls

    Returns:
        Tuple of (is_allowed, parent_id)
        - is_allowed: True if this is a thread with an allowlisted parent
        - parent_id: The parent channel ID if this is a thread, else None
    """
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bot {token}"}
            url = f"https://discord.com/api/v10/channels/{channel_id}"

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    channel_data = await response.json()
                    channel_type = channel_data.get("type")

                    # Discord thread types: 10 (GUILD_NEWS_THREAD), 11 (GUILD_PUBLIC_THREAD), 12 (GUILD_PRIVATE_THREAD)
                    if channel_type in [10, 11, 12]:
                        parent_id = channel_data.get("parent_id")
                        if parent_id and parent_id in allowed_ids:
                            logger.info(
                                f"discord_post: Thread {channel_id} has allowlisted parent {parent_id}"
                            )
                            return True, parent_id
                        else:
                            logger.info(
                                f"discord_post: Thread {channel_id} parent {parent_id} not in allowlist"
                            )
                            return False, parent_id
                    else:
                        logger.info(
                            f"discord_post: Channel {channel_id} is not a thread (type: {channel_type})"
                        )
                        return False, None
                else:
                    logger.warning(
                        f"discord_post: Failed to fetch channel info for {channel_id}: {response.status}"
                    )
                    return False, None
    except Exception as e:
        logger.error(f"discord_post: Error checking thread parent: {e}")
        return False, None


async def send_channel_message(
    client: discord.Client,
    deployment: Deployment,
    channel_id: str,
    content: str,
    files: list = None,
    reply_to: str = None,
    skip_allowlist_check: bool = False,
):
    """Send a message to a Discord channel (existing functionality)."""
    if not skip_allowlist_check:
        # Get allowed channels from deployment config
        allowed_channels = deployment.config.discord.channel_allowlist or []
        allowed_ids = [str(channel.id) for channel in allowed_channels]

        # Check if channel is directly in allowlist
        is_allowed = channel_id in allowed_ids

        # If not directly allowed, check if it's a thread with an allowlisted parent
        if not is_allowed:
            is_allowed, _ = await check_thread_parent_allowlist(
                channel_id, allowed_ids, deployment.secrets.discord.token
            )

        if not is_allowed:
            allowed_channels_info = {
                channel.note: str(channel.id) for channel in allowed_channels
            }
            raise Exception(
                f"Channel {channel_id} is not in the allowlist. Allowed channels (note: id): {allowed_channels_info}"
            )

    # Split content into chunks (Discord limit is 2000 chars)
    chunks = split_content_into_chunks(content)
    if not chunks:
        chunks = [""]  # Allow empty content if files are provided

    # Get the channel
    channel = await client.fetch_channel(int(channel_id))

    # Build message reference if replying (only for first message)
    reference = None
    if reply_to:
        reference = discord.MessageReference(
            message_id=int(reply_to), channel_id=int(channel_id)
        )

    messages = []
    last_idx = len(chunks) - 1
    for i, chunk in enumerate(chunks):
        # First message gets reply reference, last message gets files
        is_first = i == 0
        is_last = i == last_idx

        if is_first and is_last:
            # Single chunk: gets both reply reference and files
            message = await channel.send(
                content=chunk, files=files if files else None, reference=reference
            )
        elif is_first:
            # First of multiple: reply reference only
            message = await channel.send(content=chunk, reference=reference)
        elif is_last:
            # Last of multiple: files only
            message = await channel.send(content=chunk, files=files if files else None)
        else:
            # Middle chunks: plain text
            message = await channel.send(content=chunk)
        messages.append(message)

    # Build URLs - handle both guild channels and DM channels
    output = []
    for msg in messages:
        if hasattr(channel, "guild") and channel.guild:
            url = (
                f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{msg.id}"
            )
        else:
            # DM channel
            url = f"https://discord.com/channels/@me/{channel.id}/{msg.id}"
        output.append({"url": url})

    # return just the first url if messages are split
    output = output[:1]

    return {"output": output}


async def send_webhook_message(
    deployment: Deployment,
    agent: Agent,
    channel_id: str,
    content: str,
    media_urls: list = None,
    skip_allowlist_check: bool = False,
):
    """Send a message via Discord webhook (V3 deployments)."""
    from eve.s3 import get_full_url

    # Find the channel config with webhook info
    channel_config = None
    if deployment.config and deployment.config.discord:
        for ch in deployment.config.discord.channel_configs or []:
            if ch.channel_id == channel_id:
                channel_config = ch
                break

    if not channel_config:
        raise Exception(f"Channel {channel_id} not found in deployment configuration")

    # Check write access if not skipping validation
    if not skip_allowlist_check and channel_config.access != "read_write":
        raise Exception(
            f"Channel {channel_id} ({channel_config.channel_name}) is read-only. "
            "Agent can only post to channels with read_write access."
        )

    if not channel_config.webhook_id or not channel_config.webhook_token:
        raise Exception(f"No webhook configured for channel {channel_id}")

    # Split content into chunks
    chunks = split_content_into_chunks(content)
    if not chunks:
        chunks = [""]  # Allow empty if media_urls provided

    # Build webhook payload
    webhook_url = f"https://discord.com/api/v10/webhooks/{channel_config.webhook_id}/{channel_config.webhook_token}"

    # Get full URL for avatar (may be just a filename)
    avatar_url = (
        get_full_url(agent.userImage)
        if agent.userImage and not agent.userImage.startswith("http")
        else agent.userImage
    )

    messages = []
    async with aiohttp.ClientSession() as session:
        for i, chunk in enumerate(chunks):
            payload = {
                "content": chunk,
                "username": agent.name,  # Custom bot name
                "avatar_url": avatar_url,  # Custom bot avatar
                "allowed_mentions": {"parse": ["users", "roles"]},
            }

            # Add media URLs to the last chunk (webhooks don't support file uploads well)
            # So we just append URLs as text
            if i == len(chunks) - 1 and media_urls:
                media_content = "\n".join(media_urls)
                payload["content"] = (
                    f"{chunk}\n\n{media_content}" if chunk else media_content
                )

            async with session.post(webhook_url, json=payload) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    raise Exception(f"Webhook request failed: {error_text}")

                # Discord returns the message object on success
                if response.status == 200:
                    msg_data = await response.json()
                    messages.append(msg_data)

    # Build output URLs (webhook messages don't have guild_id in response, need to construct)
    guild_id = deployment.config.discord.guild_id
    output = []
    for msg in messages[:1]:  # Return just first URL if split
        url = f"https://discord.com/channels/{guild_id}/{channel_id}/{msg['id']}"
        output.append({"url": url})

    return {"output": output}
