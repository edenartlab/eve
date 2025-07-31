from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import discord


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")

    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist
    if not allowed_channels:
        raise Exception("No channels configured for this deployment")

    # Get channel ID and content from args
    channel_id = args["channel_id"]
    content = args["content"]

    # Verify the channel is in the allowlist
    if not any(str(channel.id) == channel_id for channel in allowed_channels):
        allowed_channels_info = {
            channel.note: str(channel.id) for channel in allowed_channels
        }
        raise Exception(
            f"Channel {channel_id} is not in the allowlist. Allowed channels (note: id): {allowed_channels_info}"
        )

    # Create Discord client
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Get the channel and post the message
        channel = await client.fetch_channel(int(channel_id))
        message = await channel.send(content=content)

        return {
            "output": [
                {
                    "url": f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{message.id}",
                }
            ]
        }

    finally:
        await client.close()
