from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from pydantic import BaseModel
import discord
from datetime import datetime, timedelta, timezone
import traceback


class ChannelSearchParams(BaseModel):
    channel_id: str  # The note/name of the channel to search
    message_limit: int | None = None  # Optional: number of messages to fetch
    time_window_hours: int = 24


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

    include_thread_messages = args.get("include_thread_messages", True)
    print(f"include_thread_messages: {include_thread_messages}")

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

Available channel IDs + notes:
{channel_notes}

Example queries:
"Show me recent tech support messages" -> Search in tech support channels, last 10 messages
"Get all announcements from the last 24 hours" -> Search in announcement channels, last 24 hours, no message limit
"Show me the last 5 messages from general discussion" -> Search in general channels, last 5 messages

You must return a list of ChannelSearchParams objects, each containing:
- channel_id: The ID of the channel to search (must match one of the available channel IDs)
- message_limit: Optional number of messages to fetch. If provided, returns the most recent N messages. If not provided, returns ALL messages within the time window (up to 24h max)
- time_window_hours: Time window in hours, default 24

Behavior:
- If message_limit is specified: Return the most recent N messages, going back as far as needed (or until channel start)
- If message_limit is NOT specified: Return ALL messages within the time_window_hours (up to 24h max)""".format(
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

            if channel_params.message_limit:
                # If message_limit is specified, get the most recent N messages
                # Don't set 'after' parameter - we want to go back as far as needed
                # Discord API limit is 100 messages per request
                params["limit"] = min(channel_params.message_limit, 100)
            else:
                # If no message_limit, get all messages within the time window
                # Set limit to 100 (Discord's max per request) and use time window
                after = datetime.now(timezone.utc) - timedelta(
                    hours=channel_params.time_window_hours
                )
                params["limit"] = 100  # Discord's max per request
                params["after"] = int((after.timestamp() - 1420070400) * 1000) << 22

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

            # Get thread messages if include_thread_messages is True
            if include_thread_messages:
                try:
                    # Get the guild_id from the channel data
                    guild_id = channel_data.get("guild_id")
                    if not guild_id:
                        print(f"Warning: No guild_id found for channel {channel_id}")
                        continue

                    print(f"Getting active threads for guild {guild_id}")

                    # Get all active threads for the guild
                    active_threads_response = await http.request(
                        discord.http.Route("GET", f"/guilds/{guild_id}/threads/active")
                    )

                    all_threads = active_threads_response.get("threads", [])
                    print(
                        f"Found {len(all_threads)} active threads in guild {guild_id}"
                    )

                    # Filter threads that belong to this channel
                    channel_threads = [
                        t for t in all_threads if t.get("parent_id") == str(channel_id)
                    ]
                    print(
                        f"Found {len(channel_threads)} threads for channel {channel_id}"
                    )

                    # Get messages from each thread
                    for thread in channel_threads:
                        thread_id = thread["id"]
                        thread_name = thread.get("name", "Unknown Thread")

                        try:
                            print(
                                f"Getting messages from thread {thread_id} ({thread_name})"
                            )

                            thread_params = {}
                            if channel_params.message_limit:
                                # Use same message limit for threads as main channel
                                thread_params["limit"] = min(
                                    channel_params.message_limit, 100
                                )
                            else:
                                # Use time window for threads
                                after = datetime.now(timezone.utc) - timedelta(
                                    hours=channel_params.time_window_hours
                                )
                                thread_params["limit"] = 100
                                thread_params["after"] = (
                                    int((after.timestamp() - 1420070400) * 1000) << 22
                                )

                            thread_messages = await http.logs_from(
                                int(thread_id), **thread_params
                            )
                            print(
                                f"Retrieved {len(thread_messages)} messages from thread {thread_id}"
                            )

                            # Get the thread starter message (the original message that created the thread)
                            try:
                                parent_message_id = thread.get(
                                    "id"
                                )  # In threads, the thread ID often equals the starter message ID
                                # Try to get the first message in the thread which is usually the starter
                                if (
                                    thread_messages
                                    and thread_messages[-1]["id"] == parent_message_id
                                ):
                                    # Mark the first message as the thread starter
                                    starter_msg = thread_messages[-1]
                                    messages.append(
                                        {
                                            "id": str(starter_msg["id"]),
                                            "content": starter_msg.get("content", ""),
                                            "author": starter_msg.get("author", {}).get(
                                                "username", "Unknown"
                                            ),
                                            "created_at": starter_msg.get(
                                                "timestamp", ""
                                            ),
                                            "channel_id": str(thread_id),
                                            "channel_name": f"{channel_data.get('name', 'Unknown')} > {thread_name}",
                                            "url": f"https://discord.com/channels/{guild_id}/{thread_id}/{starter_msg['id']}",
                                            "thread_parent_id": str(channel_id),
                                            "is_thread_message": True,
                                            "is_thread_starter": True,
                                        }
                                    )
                                    # Add rest of the messages (excluding the starter we already added)
                                    thread_messages = thread_messages[:-1]
                            except Exception as e:
                                print(f"Error getting thread starter message: {e}")
                                print(traceback.format_exc())

                            # Add all thread messages
                            for msg in thread_messages:
                                messages.append(
                                    {
                                        "id": str(msg["id"]),
                                        "content": msg.get("content", ""),
                                        "author": msg.get("author", {}).get(
                                            "username", "Unknown"
                                        ),
                                        "created_at": msg.get("timestamp", ""),
                                        "channel_id": str(thread_id),
                                        "channel_name": f"{channel_data.get('name', 'Unknown')} > {thread_name}",
                                        "url": f"https://discord.com/channels/{guild_id}/{thread_id}/{msg['id']}",
                                        "thread_parent_id": str(channel_id),
                                        "is_thread_message": True,
                                    }
                                )

                        except Exception as e:
                            print(f"Error processing thread {thread_id}: {e}")
                            print(traceback.format_exc())
                            continue

                except Exception as e:
                    print(f"Error getting threads for channel {channel_id}: {e}")
                    print(traceback.format_exc())

        # Sort messages by created_at timestamp to ensure reverse chronological order (newest first)
        messages.sort(key=lambda x: x["created_at"], reverse=True)

        return {"output": messages}

    finally:
        await http.close()
