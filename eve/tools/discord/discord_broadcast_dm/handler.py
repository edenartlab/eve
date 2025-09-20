import os
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set, Optional, Tuple
import discord
from bson import ObjectId

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment, Session
from eve.user import User
from eve.utils import serialize_json
from pydantic import BaseModel


def construct_agent_chat_url(agent_username: str) -> str:
    """
    Construct the Eden agent chat URL based on the environment and agent username.

    Args:
        agent_username: The username of the agent (not the ID)

    Returns:
        Properly formatted Eden agent chat URL
    """
    db = os.getenv("DB", "STAGE")
    root_url = "app.eden.art" if db == "PROD" else "staging.app.eden.art"
    return f"https://{root_url}/chat/{agent_username}"


class DiscordUser(BaseModel):
    discord_id: str
    discord_username: str
    message_count: int
    last_seen: str


async def handler(args: dict, user: str = None, agent: str = None):
    """
    Main handler for discord_broadcast_dm tool.

    Args:
        channel_id: Discord channel ID to find active users from
        instruction: What the agent should say/do in each DM
        active_days: Days to look back for activity (default 3)
        message_limit: Max messages to check per user
    """
    if not agent:
        raise Exception("Agent is required")

    agent_obj = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployment found")

    agent_owner_id = str(agent_obj.owner) if agent_obj.owner else None

    # Extract parameters
    channel_id = args["channel_id"]
    instruction = args["instruction"]
    active_days = args.get("active_days", 3)
    message_limit = args.get("message_limit", 100)
    rate_limit_delay = args.get("rate_limit_delay", 1.0)

    # Validate parameters
    if not channel_id or not instruction:
        raise Exception("channel_id and instruction are required")

    # Validate Discord snowflake format (should be numeric and reasonable length)
    try:
        channel_id_int = int(channel_id)
        if channel_id_int <= 0 or len(channel_id) < 15 or len(channel_id) > 20:
            raise ValueError("Invalid snowflake")
    except ValueError:
        raise Exception(f"channel_id must be a valid Discord snowflake, got: {channel_id}")

    if active_days <= 0 or active_days > 30:
        raise Exception("active_days must be between 1 and 30")

    if message_limit <= 0 or message_limit > 1000:
        raise Exception("message_limit must be between 1 and 1000")

    if rate_limit_delay < 0.1 or rate_limit_delay > 10.0:
        raise Exception("rate_limit_delay must be between 0.1 and 10.0 seconds")

    print(f"Starting discord_broadcast_dm for agent {agent} in channel {channel_id}")
    print(f"Looking for users active in last {active_days} days")

    try:
        # Step 1: Validate channel access
        await validate_channel_access(deployment, channel_id)

        # Step 2: Discover active users from the Discord channel
        active_users = await discover_active_users(
            deployment, channel_id, active_days, message_limit
        )
        print(f"Found {len(active_users)} active users")

        if not active_users:
            return {"output": "No active users found in the specified channel and timeframe"}

        # Step 3: Map Discord users to Eden users
        eden_users = await map_discord_to_eden_users(active_users)
        print(f"Mapped to {len(eden_users)} Eden users")

        # Step 4: Create/find DM sessions and send messages with rate limiting
        results = await orchestrate_dm_sessions(
            agent,
            deployment,
            eden_users,
            instruction,
            rate_limit_delay,
            agent_owner_id,
            agent_obj.username,
        )

        return {"output": results}

    except Exception as e:
        error_msg = f"Error in discord_broadcast_dm handler for agent {agent} in channel {channel_id}: {e}"
        print(error_msg)
        raise Exception(error_msg) from e


async def validate_channel_access(deployment: Deployment, channel_id: str):
    """
    Validate that the bot has access to the specified channel based on deployment config.
    """
    # Get allowed channels from deployment config (same as discord_search)
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

    # Check if the requested channel is in the allowed list
    allowed_channel_ids = {str(channel.id) for channel in all_channels}
    if channel_id not in allowed_channel_ids:
        available_channels = {f"{channel.note}: {channel.id}" for channel in all_channels}
        raise Exception(
            f"Channel {channel_id} is not in the configured channel access list. "
            f"Available channels: {available_channels}"
        )


async def discover_active_users(
    deployment: Deployment,
    channel_id: str,
    active_days: int,
    message_limit: int,
    include_threads: bool = True
) -> List[DiscordUser]:
    """
    Discover active users from a Discord channel within the specified timeframe.
    Includes thread messages and filters out bot users.
    """
    # Create Discord HTTP client with timeout
    http = discord.http.HTTPClient()

    try:
        await http.static_login(deployment.secrets.discord.token)
        # Check if channel exists and is accessible
        try:
            channel_data = await http.get_channel(int(channel_id))
        except discord.NotFound:
            raise Exception(f"Channel {channel_id} not found or bot lacks access")
        except discord.Forbidden:
            raise Exception(f"Bot lacks permission to access channel {channel_id}")
        except Exception as e:
            raise Exception(f"Failed to access channel {channel_id}: {str(e)}")

        if channel_data.get("type") not in [0, 5, 10, 11, 12]:  # Text channel types
            raise Exception(f"Channel {channel_id} is not a text channel (type: {channel_data.get('type')})")

        # Calculate time window
        after_time = datetime.now(timezone.utc) - timedelta(days=active_days)
        after_snowflake = int((after_time.timestamp() - 1420070400) * 1000) << 22

        # Get messages from the channel
        message_data = await http.logs_from(
            int(channel_id),
            limit=message_limit,
            after=after_snowflake
        )

        # Count user activity
        user_activity: Dict[str, Dict] = {}

        for msg in message_data:
            author = msg.get("author", {})
            author_id = author.get("id")
            author_username = author.get("username")
            timestamp = msg.get("timestamp", "")
            is_bot = author.get("bot", False)

            # Skip if missing data or is a bot
            if not author_id or not author_username or is_bot:
                continue

            if author_id not in user_activity:
                user_activity[author_id] = {
                    "username": author_username,
                    "message_count": 0,
                    "last_seen": timestamp
                }

            user_activity[author_id]["message_count"] += 1
            # Keep the most recent timestamp
            if timestamp > user_activity[author_id]["last_seen"]:
                user_activity[author_id]["last_seen"] = timestamp

        # Convert to DiscordUser objects
        active_users = []
        for user_id, data in user_activity.items():
            if data["message_count"] > 0:  # Only users with at least 1 message
                active_users.append(DiscordUser(
                    discord_id=user_id,
                    discord_username=data["username"],
                    message_count=data["message_count"],
                    last_seen=data["last_seen"]
                ))

        # Include thread messages if enabled
        if include_threads:
            try:
                thread_users = await discover_thread_users(
                    http, channel_data, active_days, min(message_limit // 2, 100)  # Limit thread messages
                )
                # Merge thread activity into main user activity
                for thread_user in thread_users:
                    existing_user = next(
                        (u for u in active_users if u.discord_id == thread_user.discord_id),
                        None
                    )
                    if existing_user:
                        existing_user.message_count += thread_user.message_count
                        if thread_user.last_seen > existing_user.last_seen:
                            existing_user.last_seen = thread_user.last_seen
                    else:
                        active_users.append(thread_user)
            except Exception as e:
                print(f"Warning: Failed to get thread messages for channel {channel_id}: {e}")

        # Sort by message count (most active first)
        active_users.sort(key=lambda x: x.message_count, reverse=True)

        return active_users

    finally:
        await http.close()


async def discover_thread_users(
    http: discord.http.HTTPClient,
    channel_data: dict,
    active_days: int,
    message_limit: int
) -> List[DiscordUser]:
    """
    Discover active users from threads in the channel.
    """
    thread_users = []

    try:
        guild_id = channel_data.get("guild_id")
        if not guild_id:
            return thread_users

        channel_id = str(channel_data["id"])

        # Get active threads for the guild
        active_threads_response = await http.request(
            discord.http.Route("GET", f"/guilds/{guild_id}/threads/active")
        )

        all_threads = active_threads_response.get("threads", [])

        # Filter threads that belong to this channel
        channel_threads = [
            t for t in all_threads if t.get("parent_id") == channel_id
        ]

        # Calculate time window using UTC
        after_time = datetime.now(timezone.utc) - timedelta(days=active_days)
        after_snowflake = int((after_time.timestamp() - 1420070400) * 1000) << 22

        # Get messages from each thread
        for thread in channel_threads:
            thread_id = thread["id"]

            try:
                thread_messages = await http.logs_from(
                    int(thread_id),
                    limit=min(message_limit, 50),  # Further limit per thread
                    after=after_snowflake
                )

                # Process thread messages
                user_activity = {}
                for msg in thread_messages:
                    author = msg.get("author", {})
                    author_id = author.get("id")
                    author_username = author.get("username")
                    timestamp = msg.get("timestamp", "")
                    is_bot = author.get("bot", False)

                    # Skip if missing data or is a bot
                    if not author_id or not author_username or is_bot:
                        continue

                    if author_id not in user_activity:
                        user_activity[author_id] = {
                            "username": author_username,
                            "message_count": 0,
                            "last_seen": timestamp
                        }

                    user_activity[author_id]["message_count"] += 1
                    if timestamp > user_activity[author_id]["last_seen"]:
                        user_activity[author_id]["last_seen"] = timestamp

                # Convert to DiscordUser objects
                for user_id, data in user_activity.items():
                    if data["message_count"] > 0:
                        thread_users.append(DiscordUser(
                            discord_id=user_id,
                            discord_username=data["username"],
                            message_count=data["message_count"],
                            last_seen=data["last_seen"]
                        ))

            except Exception as e:
                print(f"Warning: Failed to get messages from thread {thread_id}: {e}")
                continue

    except Exception as e:
        print(f"Warning: Failed to get threads: {e}")

    return thread_users


async def map_discord_to_eden_users(discord_users: List[DiscordUser]) -> List[Tuple[DiscordUser, User]]:
    """
    Map Discord users to Eden users, creating shadow users if needed.
    Returns list of (discord_user, eden_user) tuples.
    """
    mapped_users = []

    for discord_user in discord_users:
        try:
            # Use existing User.from_discord method
            eden_user = User.from_discord(
                discord_user.discord_id,
                discord_user.discord_username
            )
            mapped_users.append((discord_user, eden_user))
            print(f"Mapped Discord user {discord_user.discord_username} to Eden user {eden_user.username}")
        except Exception as e:
            error_msg = f"Failed to map Discord user {discord_user.discord_username} (ID: {discord_user.discord_id}): {e}"
            print(error_msg)
            continue

    return mapped_users


async def orchestrate_dm_sessions(
    agent: str,
    deployment: Deployment,
    eden_users: List[Tuple[DiscordUser, User]],
    instruction: str,
    rate_limit_delay: float = 1.0,
    acting_user_id: Optional[str] = None,
    agent_username: Optional[str] = None,
) -> List[Dict]:
    """
    Create or find DM sessions for each user and send personalized messages with rate limiting.
    """
    results = []
    session_tasks = []

    for discord_user, eden_user in eden_users:
        task = create_dm_session_task(
            agent,
            deployment,
            discord_user,
            eden_user,
            instruction,
            acting_user_id,
            agent_username,
        )
        session_tasks.append(task)

    # Execute DM sessions with rate limiting
    session_results = []
    for i, task in enumerate(session_tasks):
        if i > 0:  # Add delay between DM sends (except for the first one)
            await asyncio.sleep(rate_limit_delay)

        try:
            result = await task
            session_results.append(result)
        except Exception as e:
            session_results.append(e)

    for i, result in enumerate(session_results):
        discord_user, eden_user = eden_users[i]

        if isinstance(result, Exception):
            error_detail = f"DM failed for user {discord_user.discord_username} (Discord ID: {discord_user.discord_id}, Eden user: {eden_user.username}): {str(result)}"
            print(error_detail)
            results.append({
                "discord_user": discord_user.discord_username,
                "discord_id": discord_user.discord_id,
                "eden_user": eden_user.username,
                "status": "failed",
                "error": str(result)
            })
        else:
            results.append({
                "discord_user": discord_user.discord_username,
                "discord_id": discord_user.discord_id,
                "eden_user": eden_user.username,
                "status": "success",
                "session_id": result.get("session_id"),
                "dm_sent": result.get("dm_sent", False)
            })

    return results


async def create_dm_session_task(
    agent: str,
    deployment: Deployment,
    discord_user: DiscordUser,
    eden_user: User,
    instruction: str,
    acting_user_id: Optional[str] = None,
    agent_username: Optional[str] = None,
) -> Dict:
    """
    Create or reuse a DM session for a specific user and send the instruction.
    """
    # Create consistent session key for DM sessions
    session_key = f"dm_{agent}_{discord_user.discord_id}"
    print(f"Searching for existing session with key: {session_key}")

    # Check if DM session already exists (limit to 1 for efficiency)
    existing_session = Session.find_one({"session_key": session_key, "status": "active"})

    session_id = None
    if existing_session:
        session_id = str(existing_session.id)
        print(f"Found existing DM session {session_id} for user {discord_user.discord_username} (key: {session_key})")
    else:
        print(f"No existing session found for key {session_key}, will create new DM session for user {discord_user.discord_username}")

    # Get channel mentions from deployment config (similar to gateway_v2.py)
    channel_mentions = []
    if (
        deployment.config
        and deployment.config.discord
        and deployment.config.discord.channel_allowlist
    ):
        for item in deployment.config.discord.channel_allowlist:
            if item and item.id:
                channel_mentions.append(f"<#{item.id}>")

    # Build specific instruction for including channel links
    channel_instruction = ""
    if channel_mentions:
        channels_text = ", ".join(channel_mentions)
        channel_instruction = f"\n- MUST include these clickable Discord channel links in your response: {channels_text}"
    else:
        channel_instruction = "\n- Mention that they can find you in the public Discord channels you're active in."

    # Prepare session prompt request
    request_data = {
        "session_id": session_id,
        "user_id": str(eden_user.id),
        "actor_agent_ids": [agent],
        "message": {
            "role": "system",
            "content": f"""## DM Task

You are going to send a direct message to Discord user @{discord_user.discord_username}.

Context about this user:
- Discord username: {discord_user.discord_username}
- Discord ID: {discord_user.discord_id}
- Eden username: {eden_user.username}
- Last seen: {discord_user.last_seen}

Instructions:
{instruction}

Its likely this is part of a daily, recurring task, so leverage the previous conversation context to make the message more contextualized and following a linear, constructive narrative.
Important:
- Users currently cannot reply to DMs on Discord, so always include a link to the eden agent chat url: {construct_agent_chat_url(agent_username) if agent_username else f"https://app.eden.art/chat/{str(agent)}"}.{channel_instruction}
- After generating your response, use the discord_post tool to send it as a DM to this user by setting discord_user_id to {discord_user.discord_id}
""",
        },
    }

    if acting_user_id:
        request_data["acting_user_id"] = acting_user_id

    # Add session creation args if needed
    if not session_id:
        request_data["creation_args"] = {
            "owner_id": str(eden_user.id),
            "agents": [agent],
            "session_key": session_key,
            "users": [str(eden_user.id)],  # Restrict session to this user
            "platform": "discord"
        }

    # Send request to session prompt endpoint
    api_url = os.getenv("EDEN_API_URL")
    print(f"Sending session prompt request for user {discord_user.discord_username} (session_id: {session_id})")
    print(f"Request data: {request_data}")

    # Serialize request_data to properly handle ObjectId fields
    serialized_data = serialize_json(request_data)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/sessions/prompt",
            json=serialized_data,
            headers={
                "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Failed to create/prompt DM session for user {discord_user.discord_username} "
                    f"(Discord ID: {discord_user.discord_id}): HTTP {response.status} - {error_text}"
                )

            try:
                result = await response.json()
                returned_session_id = result.get("session_id")
                if not returned_session_id:
                    raise Exception(f"No session_id returned from API for user {discord_user.discord_username}")

                print(f"Session prompt completed for user {discord_user.discord_username}, returned session_id: {returned_session_id}")

                # Check if the session actually completed and sent a DM
                # The /sessions/prompt endpoint waits for completion, so if we get here the session has processed
                # We need to check if discord_post was actually called by looking at tool_calls in the result
                dm_sent = False
                tool_calls_count = 0
                discord_post_calls = 0

                if "tool_calls" in result:
                    tool_calls_count = len(result["tool_calls"])
                    print(f"Session {returned_session_id} made {tool_calls_count} tool calls")

                    for tool_call in result["tool_calls"]:
                        tool_name = tool_call.get("tool")
                        print(f"  - Tool call: {tool_name}")
                        if tool_name == "discord_post":
                            discord_post_calls += 1
                            user_id_arg = tool_call.get("args", {}).get("discord_user_id")
                            print(f"    - discord_user_id: {user_id_arg} (target: {discord_user.discord_id})")
                            if user_id_arg == discord_user.discord_id:
                                dm_sent = True
                                print(f"    - ✓ DM sent successfully to {discord_user.discord_username}")
                else:
                    print(f"Session {returned_session_id} made no tool calls")

                print(f"Final result for {discord_user.discord_username}: dm_sent={dm_sent}, discord_post_calls={discord_post_calls}")

                return {
                    "session_id": returned_session_id,
                    "dm_sent": dm_sent,
                    "user": discord_user.discord_username
                }
            except Exception as json_error:
                error_text = await response.text()
                raise Exception(
                    f"Failed to parse response for user {discord_user.discord_username}: {json_error}. "
                    f"Response: {error_text}"
                )
