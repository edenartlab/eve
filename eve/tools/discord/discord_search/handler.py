from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from pydantic import BaseModel
import discord
from datetime import datetime, timedelta, timezone
import re
import traceback
from typing import Any, Dict, List, Optional


DISCORD_EPOCH_MS = 1420070400000
MENTION_PATTERN = re.compile(r"<@!?(\d+)>")


class ChannelSearchParams(BaseModel):
    channel_id: str  # The note/name of the channel to search
    message_limit: int | None = None  # Optional: number of messages to fetch
    time_window_hours: int = 24


class DiscordSearchQuery(BaseModel):
    channels: list[ChannelSearchParams]


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
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
        messages: List[Dict[str, Any]] = []
        guild_thread_cache: Dict[str, List[Dict[str, Any]]] = {}

        for channel_params in parsed_query.channels:
            channel_messages = await _collect_channel_messages(
                http=http,
                channel_params=channel_params,
                include_thread_messages=include_thread_messages,
                channel_map=channel_map,
                guild_thread_cache=guild_thread_cache,
            )
            messages.extend(channel_messages)

        if messages:
            messages.sort(
                key=lambda msg: _parse_iso_timestamp(msg.get("created_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )

        if messages:
            messages = await _replace_user_mentions(http=http, messages=messages)

        formatted_messages = _format_output_messages(messages)
        return {"output": formatted_messages, "_skip_upload_processing": True}

    finally:
        await http.close()


def _parse_iso_timestamp(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        # Discord timestamps are ISO 8601 with Z suffix
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except Exception:
        return None


def _datetime_to_snowflake(dt: datetime) -> int:
    milliseconds = int(dt.timestamp() * 1000)
    return max(((milliseconds - DISCORD_EPOCH_MS) << 22), 0)


def _snowflake_to_datetime(snowflake_id: Optional[str | int]) -> Optional[datetime]:
    if not snowflake_id:
        return None
    try:
        snowflake_int = int(snowflake_id)
    except (TypeError, ValueError):
        return None
    timestamp_ms = (snowflake_int >> 22) + DISCORD_EPOCH_MS
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def _coerce_channel_id(channel_id_or_note: str, channel_map: Dict[str, str]) -> str:
    if not channel_id_or_note:
        raise Exception("Channel identifier is required")
    channel_id_or_note = str(channel_id_or_note)
    mapped = channel_map.get(channel_id_or_note.lower())
    return mapped or channel_id_or_note


async def _collect_channel_messages(
    http: discord.http.HTTPClient,
    channel_params: ChannelSearchParams,
    include_thread_messages: bool,
    channel_map: Dict[str, str],
    guild_thread_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    channel_id = _coerce_channel_id(channel_params.channel_id, channel_map)

    channel_data = await http.get_channel(int(channel_id))
    if channel_data.get("type") not in {0, 5, 10, 11, 12}:
        return []

    guild_id = channel_data.get("guild_id")
    channel_name = channel_data.get("name", "Unknown")

    message_limit = channel_params.message_limit
    has_message_limit = message_limit is not None
    per_request_limit = _determine_request_limit(message_limit)

    cutoff_dt: Optional[datetime]
    raw_channel_messages: List[Dict[str, Any]]

    if has_message_limit:
        print(
            f"Fetching channel {channel_name} ({channel_id}) with limit={per_request_limit}"
        )
        raw_channel_messages = await http.logs_from(
            int(channel_id), limit=per_request_limit
        )
        cutoff_dt = _oldest_timestamp(raw_channel_messages)
    else:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(
            hours=channel_params.time_window_hours
        )
        after_snowflake = _datetime_to_snowflake(cutoff_dt)
        print(
            f"Fetching channel {channel_name} ({channel_id}) after={after_snowflake} cutoff={cutoff_dt.isoformat()} limit={per_request_limit}"
        )
        raw_channel_messages = await http.logs_from(
            int(channel_id), limit=per_request_limit, after=after_snowflake
        )

    print(
        f"Retrieved {len(raw_channel_messages)} channel messages for {channel_name} ({channel_id})"
    )

    channel_messages = _transform_messages(
        raw_messages=raw_channel_messages,
        guild_id=guild_id,
        resolved_channel_id=channel_id,
        channel_name=channel_name,
        thread_parent_id=None,
    )

    # Filter to timeframe when message_limit provided
    if cutoff_dt:
        channel_messages = [
            msg for msg in channel_messages if msg["_created_at_dt"] >= cutoff_dt
        ]
        print(
            f"Channel {channel_name} ({channel_id}) retained {len(channel_messages)} messages after cutoff"
        )

    if include_thread_messages and guild_id and cutoff_dt:
        thread_messages = await _collect_thread_messages(
            http=http,
            guild_id=str(guild_id),
            parent_channel_id=str(channel_id),
            channel_name=channel_name,
            cutoff_dt=cutoff_dt,
            per_request_limit=per_request_limit,
            guild_thread_cache=guild_thread_cache,
        )
        channel_messages.extend(thread_messages)
        print(
            f"Channel {channel_name} ({channel_id}) total after threads: {len(channel_messages)}"
        )

    # Sort ascending (oldest first)
    channel_messages.sort(key=lambda msg: msg["_created_at_dt"])

    if has_message_limit and message_limit is not None:
        channel_messages = channel_messages[-message_limit:]
        print(
            f"Channel {channel_name} ({channel_id}) trimmed to final {len(channel_messages)} messages"
        )

    # Strip helper field before returning
    for msg in channel_messages:
        msg.pop("_created_at_dt", None)

    return channel_messages


def _determine_request_limit(message_limit: Optional[int]) -> int:
    if message_limit is None:
        return 100
    if message_limit <= 0:
        raise Exception("message_limit must be greater than zero")
    return min(max(message_limit, 1), 100)


def _oldest_timestamp(messages: List[Dict[str, Any]]) -> Optional[datetime]:
    oldest: Optional[datetime] = None
    for message in messages:
        current = _parse_iso_timestamp(message.get("timestamp"))
        if current and (oldest is None or current < oldest):
            oldest = current
    return oldest


def _transform_messages(
    raw_messages: List[Dict[str, Any]],
    guild_id: Optional[str],
    resolved_channel_id: str,
    channel_name: str,
    thread_parent_id: Optional[str],
) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for message in raw_messages:
        created_at = message.get("timestamp", "")
        created_at_dt = _parse_iso_timestamp(created_at)
        if not created_at_dt:
            continue

        message_channel_id = str(message.get("channel_id", resolved_channel_id))
        resolved_guild_id = guild_id or message.get("guild_id")
        guild = str(resolved_guild_id) if resolved_guild_id is not None else None
        channel_identifier = (
            str(resolved_channel_id)
            if thread_parent_id is None
            else message_channel_id
        )
        target_channel_id = (
            message_channel_id
            if thread_parent_id is not None
            else str(resolved_channel_id)
        )
        author_info = message.get("author", {})
        author_name = (
            author_info.get("global_name")
            or author_info.get("username")
            or "Unknown"
        )
        message_id = str(message.get("id"))

        message_url = None
        if guild and channel_identifier and message_id:
            message_url = (
                f"https://discord.com/channels/{guild}/{channel_identifier}/{message_id}"
            )

        transformed.append(
            {
                "id": message_id,
                "content": message.get("content", ""),
                "author": author_name,
                "created_at": created_at,
                "channel_id": str(target_channel_id),
                "channel_name": channel_name,
                "guild_id": guild,
                "url": message_url,
                **(
                    {
                        "thread_parent_id": str(thread_parent_id),
                        "is_thread_message": True,
                    }
                    if thread_parent_id is not None
                    else {}
                ),
                "_created_at_dt": created_at_dt,
            }
        )
    return transformed


def _build_message_url(
    guild_id: Optional[str], channel_id: Optional[str], message_id: Optional[str]
) -> Optional[str]:
    if not guild_id or not channel_id or not message_id:
        return None
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _format_timestamp_for_output(timestamp: Optional[str]) -> Optional[str]:
    dt = _parse_iso_timestamp(timestamp) if timestamp else None
    if not dt:
        return timestamp
    return dt.strftime("%m-%d %H:%M")


def _format_output_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    include_url = True
    include_channel_name = True

    for message in messages:
        formatted_timestamp = _format_timestamp_for_output(message.get("created_at"))

        entry: Dict[str, Any] = {
            "content": message.get("content"),
            "author": message.get("author"),
            "created_at": formatted_timestamp,
        }

        if include_channel_name:
            entry["channel_name"] = message.get("channel_name") or "Unknown"

        if include_url:
            url = _build_message_url(
                guild_id=message.get("guild_id"),
                channel_id=message.get("channel_id"),
                message_id=message.get("id"),
            )
            if url and isinstance(url, str) and url.startswith("http"):
                entry["url"] = url
            elif message.get("url"):
                entry["url"] = message["url"]

        formatted.append(entry)
        include_channel_name = False
        include_url = False

    return formatted


async def _collect_thread_messages(
    http: discord.http.HTTPClient,
    guild_id: str,
    parent_channel_id: str,
    channel_name: str,
    cutoff_dt: datetime,
    per_request_limit: int,
    guild_thread_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if guild_id not in guild_thread_cache:
        response = await http.request(
            discord.http.Route("GET", f"/guilds/{guild_id}/threads/active")
        )
        guild_thread_cache[guild_id] = response.get("threads", [])

    all_threads = guild_thread_cache.get(guild_id, [])
    relevant_threads: List[Dict[str, Any]] = []

    for thread in all_threads:
        if thread.get("parent_id") != parent_channel_id:
            continue

        last_message_id = thread.get("last_message_id") or thread.get("id")
        last_activity = _snowflake_to_datetime(last_message_id)
        if not last_activity or last_activity < cutoff_dt:
            continue
        relevant_threads.append(thread)

    print(
        f"Channel {parent_channel_id} has {len(relevant_threads)} active threads within cutoff"
    )

    thread_messages: List[Dict[str, Any]] = []

    for thread in relevant_threads:
        thread_id = thread["id"]
        thread_name = thread.get("name", "Unknown Thread")

        try:
            print(
                f"Fetching thread {thread_name} ({thread_id}) for parent {parent_channel_id} limit={per_request_limit}"
            )
            raw_thread_messages = await http.logs_from(
                int(thread_id), limit=per_request_limit
            )
        except Exception as exc:
            print(f"Error processing thread {thread_id}: {exc}")
            print(traceback.format_exc())
            continue

        thread_channel_name = f"{channel_name} > {thread_name}"
        transformed = _transform_messages(
            raw_messages=raw_thread_messages,
            guild_id=guild_id,
            resolved_channel_id=thread_id,
            channel_name=thread_channel_name,
            thread_parent_id=parent_channel_id,
        )

        filtered = [
            msg for msg in transformed if msg["_created_at_dt"] >= cutoff_dt
        ]
        print(
            f"Retrieved {len(filtered)} messages from thread {thread_name} ({thread_id}) within cutoff"
        )
        thread_messages.extend(filtered)

    return thread_messages


async def _replace_user_mentions(
    http: discord.http.HTTPClient, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    user_ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not content:
            continue
        for match in MENTION_PATTERN.finditer(content):
            user_ids.add(match.group(1))

    if not user_ids:
        return messages

    username_cache: Dict[str, str] = {}
    for user_id in user_ids:
        try:
            user = await http.get_user(int(user_id))
        except Exception as exc:
            print(f"Failed to resolve user {user_id}: {exc}")
            continue
        if user:
            username = user.get("global_name") or user.get("username")
            if username:
                username_cache[user_id] = username

    def replace(content: str) -> str:
        def repl(match: re.Match[str]) -> str:
            target_id = match.group(1)
            username = username_cache.get(target_id)
            return f"@{username}" if username else match.group(0)

        return MENTION_PATTERN.sub(repl, content)

    for message in messages:
        content = message.get("content")
        if content:
            message["content"] = replace(content)

    return messages
