from eve.user import User
from ....deploy import Deployment
from ....agent import Agent
import discord
from pydantic import BaseModel


class PostMessageParams(BaseModel):
    channel_id: str  # The Discord channel ID to post to
    content: str  # The message content to post


async def handler(args: dict, user: User, agent: Agent):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")

    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist
    if not allowed_channels:
        raise Exception("No channels configured for this deployment")

    # Parse the post parameters
    params = PostMessageParams(**args.get("params", {}))

    # Verify the channel is in the allowlist
    channel_id = params.channel_id
    if not any(str(channel.id) == channel_id for channel in allowed_channels):
        raise Exception(f"Channel {channel_id} is not in the allowlist")

    # Create Discord client
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Get the channel and post the message
        channel = await client.fetch_channel(int(channel_id))
        message = await channel.send(content=params.content)

        return {
            "output": {
                "message_id": str(message.id),
                "channel_id": str(channel.id),
                "content": message.content,
                "url": f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{message.id}",
            }
        }

    finally:
        await client.close()
