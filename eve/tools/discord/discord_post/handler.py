from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import discord
import aiohttp
import io
import os


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")

    # Get parameters
    channel_id = args.get("channel_id")
    discord_user_id = args.get("discord_user_id")
    content = args["content"]
    media_urls = args.get("media_urls", [])

    # Validate parameters
    if not discord_user_id and not channel_id:
        raise Exception("Either channel_id or discord_user_id must be provided")

    if not content or not content.strip():
        raise Exception("Content cannot be empty")

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
            return await send_channel_message(client, deployment, channel_id, content, files)

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
                        filename = os.path.basename(url.split('?')[0])
                        if not filename or '.' not in filename:
                            # Determine extension from content-type
                            content_type = response.headers.get('content-type', '')
                            if 'image' in content_type:
                                ext = content_type.split('/')[-1]
                                filename = f"image.{ext}"
                            elif 'video' in content_type:
                                ext = content_type.split('/')[-1]
                                filename = f"video.{ext}"
                            else:
                                filename = "attachment"

                        # Read file content
                        file_data = await response.read()
                        file_obj = discord.File(io.BytesIO(file_data), filename=filename)
                        files.append(file_obj)
            except Exception as e:
                # If any file fails to download, return empty list to trigger fallback
                print(f"Failed to download media from {url}: {str(e)}")
                return []

    return files


async def send_dm(client: discord.Client, discord_user_id: str, content: str, files: list = None):
    """Send a direct message to a Discord user."""
    try:
        # Get the user object
        user = await client.fetch_user(int(discord_user_id))

        # Truncate content to 2000 characters (Discord limit)
        content = content[:2000]

        # Send DM with files if provided
        if files:
            message = await user.send(content=content, files=files)
        else:
            message = await user.send(content=content)

        # Return with dummy URL since DMs don't have public URLs
        return {
            "output": [
                {
                    "url": f"https://discord.com/channels/@me/{user.dm_channel.id}/{message.id}",
                }
            ]
        }

    except discord.Forbidden:
        # User has DMs disabled or bot is blocked
        raise Exception(f"Cannot send DM to user {discord_user_id}: DMs disabled or bot blocked")
    except discord.NotFound:
        raise Exception(f"User {discord_user_id} not found")
    except Exception as e:
        raise Exception(f"Failed to send DM to user {discord_user_id}: {str(e)}")


async def send_channel_message(client: discord.Client, deployment: Deployment, channel_id: str, content: str, files: list = None):
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

    # Truncate content to 2000 characters (Discord limit)
    content = content[:2000]

    # Get the channel and post the message with files if provided
    channel = await client.fetch_channel(int(channel_id))
    if files:
        message = await channel.send(content=content, files=files)
    else:
        message = await channel.send(content=content)

    return {
        "output": [
            {
                "url": f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{message.id}",
            }
        ]
    }
