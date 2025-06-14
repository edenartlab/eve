from eve.agent.session.models import ChatMessage, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from ....deploy import Deployment
from ....agent import Agent
from pydantic import BaseModel
import discord
from datetime import datetime, timedelta
from typing import Optional


class ChannelSearchParams(BaseModel):
    channel_note: str  # The note/name of the channel to search
    message_limit: Optional[int] = None
    time_window_hours: Optional[int] = None


class DiscordSearchQuery(BaseModel):
    channels: list[ChannelSearchParams]


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid discord deployments found")

    query = args.get("query")
    if not query:
        raise Exception("Query parameter is required")

    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist or []
    read_access_channels = deployment.config.discord.read_access_channels or []

    # Combine and deduplicate channels by ID
    seen_ids = set()
    all_channels = []
    for channel in allowed_channels + read_access_channels:
        if channel.id not in seen_ids:
            seen_ids.add(channel.id)
            all_channels.append(channel)
    if not all_channels:
        raise Exception("No channels configured for this deployment")

    # Create a mapping of channel notes to their IDs
    channel_map = {str(channel.note).lower(): channel.id for channel in all_channels}

    # Use LLM to parse the search query and determine search parameters
    system_message = """You are a Discord search query parser. Your task is to:
1. Analyze the query to determine which channels to search and their specific parameters
2. For each channel, determine if we should fetch a specific number of messages or use a time window
3. Return a structured query object with a list of channels and their search parameters

Available channel notes:
{channel_notes}

Example queries:
"Show me recent tech support messages" -> Search in tech support channels, last 10 messages
"Get all announcements from the last 24 hours" -> Search in announcement channels, last 24 hours
"Show me the last 5 messages from general discussion" -> Search in general channels, last 5 messages

You must return a list of ChannelSearchParams objects, each containing:
- channel_note: The note/name of the channel to search (must match one of the available channel notes)
- message_limit: Optional number of messages to fetch
- time_window_hours: Optional time window in hours


At least one of message_limit or time_window_hours must be specified for each channel.""".format(
        channel_notes="\n".join(f"- {note}" for note in channel_map.keys())
    )

    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=f"Parse this Discord search query: {query}"),
    ]

    parsed_query = await async_prompt(
        context=LLMContext(
            messages=messages,
            metadata=ToolMetadataBuilder(
                tool_name="discord_search",
                user_id=args.get("user"),
                agent_id=args.get("agent"),
            )(),
        ),
    )

    # Create Discord client
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Get messages from relevant channels
        messages = []
        for channel_params in parsed_query.channels:
            # Get the channel ID from the note
            channel_id = channel_map.get(channel_params.channel_note.lower())
            if not channel_id:
                allowed_notes = list(channel_map.keys())
                raise Exception(
                    f"Channel note '{channel_params.channel_note}' not found. Available channel notes: {allowed_notes}"
                )

            try:
                channel = await client.fetch_channel(int(channel_id))

                # Determine time window if specified
                after = None
                if channel_params.time_window_hours:
                    after = datetime.utcnow() - timedelta(
                        hours=channel_params.time_window_hours
                    )

                # Get messages based on parsed parameters
                async for message in channel.history(
                    limit=channel_params.message_limit, after=after
                ):
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
