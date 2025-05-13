from ....deploy import Deployment
from ....agent import Agent
import discord


async def handler(args: dict):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid discord deployments found")

    query = args.get("query")
    if not query:
        raise Exception("Query parameter is required")

    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist
    if not allowed_channels:
        raise Exception("No channels configured for this deployment")

    # Create Discord client
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Get messages from relevant channels
        messages = []
        for channel_info in allowed_channels:
            channel_id = channel_info.id
            channel_note = channel_info.note
            channel_note_str = str(channel_note).lower()

            # Skip channels that don't match the query in their note
            if query.lower() not in channel_note_str:
                continue

            try:
                channel = await client.fetch_channel(int(channel_id))

                # Get last 10 messages or messages from last 24 hours
                async for message in channel.history(limit=10):
                    messages.append(
                        {
                            "id": str(message.id),
                            "content": message.content,
                            "author": str(message.author),
                            "created_at": message.created_at.isoformat(),
                            "channel_id": str(channel.id),
                            "channel_name": channel.name,
                            "url": f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{message.id}",
                        }
                    )
            except Exception as e:
                print(f"Error fetching messages from channel {channel_id}: {e}")
                continue

        return {"output": messages}

    finally:
        await client.close()
