import io
import os
from typing import List

import aiohttp
import discord
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.deployments.discord_gateway import convert_usernames_to_discord_mentions
from eve.agent.session.models import Deployment
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

    # Create Discord client
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
                client, deployment, channel_id, content, files, reply_to
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
                for msg in messages
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


async def send_channel_message(
    client: discord.Client,
    deployment: Deployment,
    channel_id: str,
    content: str,
    files: list = None,
    reply_to: str = None,
):
    """Send a message to a Discord channel (existing functionality)."""
    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist
    if not allowed_channels:
        raise Exception("No channels configured for this deployment")

    # Verify the channel is in the allowlist
    if not any(str(channel.id) == channel_id for channel in allowed_channels):
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

    return {"output": output}
