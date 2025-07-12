import traceback
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
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
        channel_notes="\n".join(f"{id}: {note}" for id, note in channel_map.items())
    )

    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=f"Parse this Discord search query: {query}"),
    ]

    response = await async_prompt(
        context=LLMContext(
            messages=messages,
            metadata=ToolMetadataBuilder(
                tool_name="discord_search",
                user_id=args.get("user"),
                agent_id=args.get("agent"),
            )(),
            config=LLMConfig(
                response_format=DiscordSearchQuery,
            ),
        ),
    )
    parsed_query = DiscordSearchQuery.model_validate_json(response.content)

    # Create Discord HTTP client directly
    http = discord.http.HTTPClient()
    await http.static_login(deployment.secrets.discord.token)

    try:
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
                # Get channel data
                channel_data = await http.get_channel(int(channel_id))

                # Check if channel supports messages (text channels)
                if channel_data.get("type") not in [
                    0,
                    5,
                    10,
                    11,
                    12,
                ]:  # Text channel types
                    print(
                        f"***debug*** Channel {channel_id} ({channel_data.get('name', 'Unknown')}) does not support message history"
                    )
                    continue

                # Determine time window if specified
                after = None
                if channel_params.time_window_hours:
                    after = datetime.utcnow() - timedelta(
                        hours=channel_params.time_window_hours
                    )

                # Get messages using HTTP client
                params = {}
                if channel_params.message_limit:
                    params["limit"] = channel_params.message_limit
                if after:
                    # Convert datetime to snowflake ID
                    after_snowflake = int((after.timestamp() - 1420070400) * 1000) << 22
                    params["after"] = after_snowflake

                message_data = await http.logs_from(int(channel_id), **params)

                for msg in message_data:
                    messages.append(
                        {
                            "id": str(msg["id"]),
                            "content": msg.get("content", ""),
                            "author": msg.get("author", {}).get("username", "Unknown"),
                            "created_at": msg.get("timestamp", ""),
                            "channel_id": str(channel_id),
                            "channel_name": channel_data.get("name", "Unknown"),
                            "url": f"https://discord.com/channels/{channel_data.get('guild_id', 'Unknown')}/{channel_id}/{msg['id']}",
                        }
                    )
            except Exception as e:
                print(
                    f"***debug*** Error fetching messages from channel {channel_id}: {e}"
                )
                print(traceback.format_exc())
                continue

        return {"output": messages}

    finally:
        await http.close()
