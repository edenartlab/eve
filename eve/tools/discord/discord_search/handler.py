from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from pydantic import BaseModel
import discord

# Print the version of discord.py:
print("discord.py version: ", discord.__version__)
from datetime import datetime, timedelta

# Configuration constants
DEFAULT_MESSAGE_LIMIT = 25
DEFAULT_TIME_WINDOW_HOURS = 48
MAX_MESSAGE_LIMIT_FALLBACK = 100
MAX_TIME_CUTOFF_HOURS = 24 * 7  # 7 days

class ChannelSearchParams(BaseModel):
    channel_id: str  # The note/name of the channel to search
    message_limit: int | None = DEFAULT_MESSAGE_LIMIT  # None means no limit
    time_window_hours: int | None = DEFAULT_TIME_WINDOW_HOURS  # None means no time limit

class DiscordSearchQuery(BaseModel):
    channels: list[ChannelSearchParams]

async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
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
    channel_notes = "\n".join(f"{id}: {note}" for id, note in channel_map.items())

    system_message = f"""You are a Discord search query parser. Your task is to:
1. Analyze the query to determine which channel(s) to search and the specific parameters
2. Determine when to apply message limits, time limits, both, or neither based on user intent
3. Return a structured query object with a list of channels and their search parameters

Available channel IDs + notes:
{channel_notes}

IMPORTANT PARSING RULES:
- If user specifies "last X messages" or "X messages", set message_limit=X and time_window_hours=null
- If user specifies "from the past X hours/days" or "all messages from timeframe", set time_window_hours=X and message_limit=null
- If user says "all messages" without timeframe, set both message_limit=null and time_window_hours=null
- If user gives both constraints ("last 10 messages from past 24h"), set both limits
- If user is vague ("recent messages"), use defaults: message_limit={DEFAULT_MESSAGE_LIMIT}, time_window_hours={DEFAULT_TIME_WINDOW_HOURS}

Example queries and their parsing:
"Get the last 10 messages from research channel" -> message_limit=10, time_window_hours=null
"Grab all messages from research channel from the past 24h" -> message_limit=null, time_window_hours=24
"Show me all messages from announcements" -> message_limit=null, time_window_hours=null
"Get recent support messages" -> message_limit={DEFAULT_MESSAGE_LIMIT}, time_window_hours={DEFAULT_TIME_WINDOW_HOURS} (defaults)
"Last 5 messages from general in past hour" -> message_limit=5, time_window_hours=1

You must return a list of ChannelSearchParams objects (one per channel to search), each containing:
- channel_id: The ID of the channel to search (must match one of the available channel IDs)
- message_limit: Number of messages to fetch, or null for no limit
- time_window_hours: Time window in hours, or null for no time limit

Set values to null when the user's intent is to ignore that constraint."""

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

            print("Querying channel: ", channel_params.channel_id)
            print("Message limit: ", channel_params.message_limit)
            print("Time window hours: ", channel_params.time_window_hours)
            print("--------------------------------")

            # Get the channel ID from the note
            channel_id = channel_params.channel_id
            if not channel_id:
                allowed_ids = list(channel_map.keys())
                raise Exception(
                    f"Channel note '{channel_params.channel_id}' not found. Available channel notes: {allowed_ids}"
                )

            # Get channel data
            channel_data = await http.get_channel(int(channel_id))

            # Check if channel supports messages (text channels)
            if channel_data.get("type") not in [
                0,
                5,
                10,
                11,
                12,
            ]:  # Text channel types=
                continue

            # Get messages using HTTP client
            params = {}
            
            # When message limit is specified, we need to fetch more messages than requested
            # to ensure we get the most recent ones, then filter afterwards
            params["limit"] = MAX_MESSAGE_LIMIT_FALLBACK
            
            # Set time window if specified, or apply max cutoff of 7 days when no limit
            if channel_params.time_window_hours is not None:
                after = datetime.utcnow() - timedelta(
                    hours=channel_params.time_window_hours
                )
                params["after"] = int((after.timestamp() - 1420070400) * 1000) << 22
            else:
                # Apply max cutoff when no time limit specified
                after = datetime.utcnow() - timedelta(hours=MAX_TIME_CUTOFF_HOURS)
                params["after"] = int((after.timestamp() - 1420070400) * 1000) << 22

            message_data = await http.logs_from(int(channel_id), **params)

            # Convert messages to our format
            channel_messages = []
            for msg in message_data:
                channel_messages.append(
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
            
            # If message limit is specified, sort by timestamp (newest first) and take the most recent N messages
            if channel_params.message_limit is not None:
                # Sort by created_at timestamp in descending order (newest first)
                channel_messages.sort(key=lambda x: x["created_at"], reverse=True)
                # Take only the most recent N messages
                channel_messages = channel_messages[:channel_params.message_limit]
            
            # Add the filtered messages to the overall messages list
            messages.extend(channel_messages)

        # Sort messages by created_at timestamp to ensure chronological order (oldest first)
        messages.sort(key=lambda x: x["created_at"])

        return {"output": messages}

    finally:
        await http.close()
