"""
Discord Gateway Helper Models and Functions

Provides MongoDB models for tracking Discord messages and helper functions
for message backfilling and media handling.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from bson import ObjectId
from loguru import logger
from pydantic import Field

from eve.mongo import Collection, Document
from eve.s3 import upload_file_from_url

# ============================================================================
# MONGODB MODELS
# ============================================================================


@Collection("discord_messages")
class DiscordMessage(Document):
    """Tracks Discord messages for deduplication and session linkage."""

    discord_message_id: str  # Discord message ID (stored separately, _id is ObjectId)
    channel_id: str
    guild_id: Optional[str] = None
    author_id: str
    author_username: str
    content: str
    timestamp: datetime
    attachments: Optional[List[Dict[str, Any]]] = None
    embeds: Optional[List[Dict[str, Any]]] = None
    mentions: Optional[List[Dict[str, Any]]] = None
    referenced_message_id: Optional[str] = None  # For replies
    # Linkage to Eve
    session_id: Optional[List[ObjectId]] = Field(
        default_factory=list
    )  # Eve session IDs (multiple sessions can share same message)
    eve_message_id: Optional[ObjectId] = None  # Single shared ChatMessage
    processed: bool = False
    first_seen_at: datetime = None
    last_seen_at: datetime = None
    # Optional attribution for webhook-sent messages
    source_agent_id: Optional[str] = None
    source_deployment_id: Optional[str] = None

    def __init__(self, **data):
        if "first_seen_at" not in data:
            data["first_seen_at"] = datetime.now(timezone.utc)
        if "last_seen_at" not in data:
            data["last_seen_at"] = datetime.now(timezone.utc)
        super().__init__(**data)


@Collection("discord_guilds")
class DiscordGuild(Document):
    """Tracks Discord guilds (servers) that a bot has access to."""

    deployment_id: ObjectId  # Links to deployments2 collection
    guild_id: str  # Discord guild snowflake ID
    name: str  # Guild name
    icon: Optional[str] = None  # Icon URL
    owner: bool = False  # Whether bot owner is guild owner
    permissions: Optional[str] = None  # Bot permissions in guild
    member_count: Optional[int] = None  # Approximate member count
    channels: List[Dict[str, Any]] = []  # Embedded channel list
    last_refreshed_at: datetime = None

    def __init__(self, **data):
        if "last_refreshed_at" not in data:
            data["last_refreshed_at"] = datetime.now(timezone.utc)
        if "channels" not in data:
            data["channels"] = []
        super().__init__(**data)


@Collection("discord_channels")
class DiscordChannel(Document):
    """Tracks Discord channels within guilds."""

    deployment_id: ObjectId  # Links to deployments2 collection
    guild_id: str  # Parent guild ID
    channel_id: str  # Discord channel snowflake ID
    name: str  # Channel name
    type: int  # Channel type (0=text, 2=voice, 4=category, etc.)
    type_name: str  # Human-readable type name
    category_id: Optional[str] = None  # Parent category ID
    category_name: Optional[str] = None  # Parent category name
    position: int = 0  # Channel position in list
    last_refreshed_at: datetime = None

    def __init__(self, **data):
        if "last_refreshed_at" not in data:
            data["last_refreshed_at"] = datetime.now(timezone.utc)
        super().__init__(**data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def fetch_discord_channel_history(
    channel_id: str,
    token: str,
    limit: int = 100,
    before: Optional[str] = None,
    after: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch message history from Discord REST API.

    Args:
        channel_id: Discord channel ID
        token: Bot token for authentication
        limit: Max messages to fetch (max 100 per request)
        before: Get messages before this message ID
        after: Get messages after this message ID

    Returns:
        List of message objects in chronological order (oldest first)
    """
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    params = {"limit": min(limit, 100)}

    if before:
        params["before"] = before
    if after:
        params["after"] = after

    messages = []

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(
                    f"Failed to fetch Discord channel history: {response.status} - {error_text}"
                )
                return []

            data = await response.json()
            messages = data

    # Discord returns newest first, reverse for chronological order
    messages.reverse()
    return messages


async def fetch_discord_channel_history_full(
    channel_id: str,
    token: str,
    max_messages: int = 200,
    max_days: int = 90,
) -> List[Dict[str, Any]]:
    """
    Fetch full channel history with pagination, respecting limits.

    Args:
        channel_id: Discord channel ID
        token: Bot token
        max_messages: Maximum total messages to fetch
        max_days: Maximum age of messages in days

    Returns:
        List of messages in chronological order (oldest first)
    """
    from datetime import timedelta

    cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_days)
    all_messages = []
    before_id = None

    while len(all_messages) < max_messages:
        batch_size = min(100, max_messages - len(all_messages))
        messages = await fetch_discord_channel_history(
            channel_id=channel_id,
            token=token,
            limit=batch_size,
            before=before_id,
        )

        if not messages:
            break

        # Filter by age and add to results
        for msg in messages:
            msg_time = parse_discord_timestamp(msg.get("timestamp"))
            if msg_time and msg_time < cutoff_time:
                # Message is too old, we've hit the cutoff
                # Since messages are in chronological order (oldest first after reverse),
                # we need to handle this differently
                continue
            all_messages.append(msg)

        # Get the oldest message ID for next pagination
        # Since we reversed, first message in batch is oldest
        if messages:
            # Get the ID of the oldest message we received (before reverse it was last)
            # After reverse, first is oldest - but we paginate backwards, so use original order
            # Actually, fetch returns newest first, then we reverse
            # So to get "before" the oldest we have, we need the first ID after reverse
            before_id = messages[0]["id"]

        # If we got fewer than requested, we've hit the beginning
        if len(messages) < batch_size:
            break

    # Final filter by cutoff time
    all_messages = [
        msg
        for msg in all_messages
        if parse_discord_timestamp(msg.get("timestamp"))
        and parse_discord_timestamp(msg.get("timestamp")) >= cutoff_time
    ]

    return all_messages


def parse_discord_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """Parse Discord ISO timestamp to datetime."""
    if not timestamp_str:
        return None
    try:
        # Discord uses ISO format with optional microseconds
        # Example: 2024-01-15T12:34:56.789000+00:00
        return datetime.fromisoformat(timestamp_str.replace("+00:00", "+00:00"))
    except (ValueError, AttributeError):
        try:
            # Fallback for different formats
            from dateutil import parser

            return parser.parse(timestamp_str)
        except Exception:
            return None


def unpack_discord_message(
    message_data: Dict[str, Any],
) -> tuple[str, str, str, str, List[str], datetime, Optional[str]]:
    """
    Extract key components from a Discord message.

    Returns:
        Tuple of (message_id, author_id, author_username, content, media_urls, timestamp, reply_to_id)
    """
    message_id = message_data.get("id", "")
    author = message_data.get("author", {})
    author_id = author.get("id", "")
    author_username = author.get("username", "Unknown")
    content = message_data.get("content", "")
    timestamp = parse_discord_timestamp(message_data.get("timestamp"))

    # Extract attachment URLs
    media_urls = []
    for attachment in message_data.get("attachments", []):
        url = attachment.get("url") or attachment.get("proxy_url")
        if url:
            media_urls.append(url)

    # Extract embed images/videos
    for embed in message_data.get("embeds", []):
        if embed.get("image", {}).get("url"):
            media_urls.append(embed["image"]["url"])
        if embed.get("video", {}).get("url"):
            media_urls.append(embed["video"]["url"])
        if embed.get("thumbnail", {}).get("url"):
            media_urls.append(embed["thumbnail"]["url"])

    # Get reply reference
    reply_to_id = None
    if message_data.get("referenced_message"):
        reply_to_id = message_data["referenced_message"].get("id")
    elif message_data.get("message_reference"):
        reply_to_id = message_data["message_reference"].get("message_id")

    return (
        message_id,
        author_id,
        author_username,
        content,
        media_urls,
        timestamp,
        reply_to_id,
    )


async def upload_discord_media_to_s3(media_urls: List[str]) -> List[str]:
    """
    Upload Discord media URLs to S3.

    Args:
        media_urls: List of Discord CDN URLs

    Returns:
        List of S3 URLs
    """
    s3_urls = []

    for url in media_urls:
        try:
            s3_url, _ = upload_file_from_url(url)
            if s3_url:
                s3_urls.append(s3_url)
            else:
                # Fallback to original URL if upload fails
                s3_urls.append(url)
        except Exception as e:
            logger.error(f"Failed to upload Discord media to S3: {url} - {e}")
            s3_urls.append(url)  # Fallback to original URL

    return s3_urls


def get_or_create_discord_message(
    message_data: Dict[str, Any],
    session_id: Optional[ObjectId] = None,
    eve_message_id: Optional[ObjectId] = None,
):
    """
    Get existing DiscordMessage or create a new one.

    Args:
        message_data: Raw Discord message data
        session_id: Eve session ID to add to the session_id array
        eve_message_id: Eve ChatMessage ID (single, shared across sessions)

    Returns:
        Tuple of (DiscordMessage, existing_ChatMessage or None)
        - If DiscordMessage already had an eve_message_id, returns the existing ChatMessage
        - Callers should add their session to the existing ChatMessage instead of creating new
    """
    from eve.agent.session.models import ChatMessage

    discord_msg_id = message_data.get("id")

    # Try to find existing by Discord message ID
    existing = DiscordMessage.find_one({"discord_message_id": discord_msg_id})
    if existing:
        # Update: add session to array, set eve_message_id if not already set
        update_ops = {"$set": {"last_seen_at": datetime.now(timezone.utc)}}
        if session_id:
            update_ops.setdefault("$addToSet", {})["session_id"] = session_id
        if eve_message_id and not existing.eve_message_id:
            update_ops["$set"]["eve_message_id"] = eve_message_id

        DiscordMessage.get_collection().update_one({"_id": existing.id}, update_ops)

        # Return existing ChatMessage if any
        existing_chat_msg = None
        if existing.eve_message_id:
            existing_chat_msg = ChatMessage.from_mongo(existing.eve_message_id)

        return existing, existing_chat_msg

    # Create new
    (
        msg_id,
        author_id,
        author_username,
        content,
        media_urls,
        timestamp,
        reply_to_id,
    ) = unpack_discord_message(message_data)

    discord_msg = DiscordMessage(
        discord_message_id=msg_id,
        channel_id=message_data.get("channel_id", ""),
        guild_id=message_data.get("guild_id"),
        author_id=author_id,
        author_username=author_username,
        content=content,
        timestamp=timestamp or datetime.now(timezone.utc),
        attachments=message_data.get("attachments"),
        embeds=message_data.get("embeds"),
        mentions=message_data.get("mentions"),
        referenced_message_id=reply_to_id,
        session_id=[session_id] if session_id else [],
        eve_message_id=eve_message_id,
    )
    discord_msg.save()
    return discord_msg, None


# ============================================================================
# MENTION CONVERSION FUNCTIONS
# ============================================================================

# Regex pattern for Discord user mentions: <@123456789> or <@!123456789>
DISCORD_MENTION_PATTERN = re.compile(r"<@!?(\d+)>")


def convert_discord_mentions_to_usernames(
    content: str,
    mentions_data: Optional[List[Dict[str, Any]]] = None,
    bot_discord_id: Optional[str] = None,
    bot_name: Optional[str] = None,
) -> str:
    """
    Convert Discord mention format (<@discord_id>) to readable @username format.

    This makes messages readable to the LLM by replacing Discord's mention syntax
    with human-readable usernames.

    Args:
        content: Message content with Discord mentions
        mentions_data: List of mentioned user objects from Discord API
        bot_discord_id: The bot's Discord application ID (to convert self-mentions)
        bot_name: The bot/agent name to use for self-mentions

    Returns:
        Content with mentions converted to @username format

    Example:
        "hey <@1258028681138540626> say my name" -> "hey chiba say my name"
    """
    if not content:
        return content

    # Build a mapping of discord_id -> username from mentions data
    mention_map = {}
    if mentions_data:
        for mention in mentions_data:
            discord_id = mention.get("id")
            username = mention.get("username")
            if discord_id and username:
                mention_map[discord_id] = username

    # Add bot mapping if provided
    if bot_discord_id and bot_name:
        mention_map[bot_discord_id] = bot_name

    def replace_mention(match):
        discord_id = match.group(1)

        # First check our mention map (from Discord API data)
        if discord_id in mention_map:
            return mention_map[discord_id]

        # Try to look up user from database
        from eve.user import User

        try:
            user = User.find_one({"discordId": discord_id})
            if user and user.discordUsername:
                return user.discordUsername
            elif user and user.username:
                return user.username
        except Exception:
            pass

        # Fallback: keep original mention but log it
        logger.debug(f"Could not resolve Discord mention for ID: {discord_id}")
        return match.group(0)

    return DISCORD_MENTION_PATTERN.sub(replace_mention, content)


def convert_usernames_to_discord_mentions(content: str) -> str:
    """
    Convert @username format back to Discord mention format (<@discord_id>).

    This converts human-readable usernames in agent responses back to proper
    Discord mentions before posting.

    Args:
        content: Message content with @username mentions

    Returns:
        Content with usernames converted to Discord mention format

    Example:
        "hey @chiba how are you" -> "hey <@1258028681138540626> how are you"
    """
    if not content:
        return content

    from eve.user import User

    # Pattern to match @username (word characters, allowing underscores and numbers)
    # Be careful not to match email addresses or URLs
    username_pattern = re.compile(r"(?<![/\w])@(\w+)(?!\.\w)")

    def replace_username(match):
        username = match.group(1)

        # Try to find user by discordUsername first
        try:
            user = User.find_one({"discordUsername": username})
            if user and user.discordId:
                return f"<@{user.discordId}>"

            # Try case-insensitive match
            user = User.find_one(
                {
                    "discordUsername": {
                        "$regex": f"^{re.escape(username)}$",
                        "$options": "i",
                    }
                }
            )
            if user and user.discordId:
                return f"<@{user.discordId}>"
        except Exception as e:
            logger.debug(f"Error looking up user {username}: {e}")

        # Fallback: keep as @username (won't ping but readable)
        return match.group(0)

    return username_pattern.sub(replace_username, content)


# ============================================================================
# GUILD/CHANNEL REFRESH FUNCTIONS
# ============================================================================


async def refresh_discord_guilds_and_channels(deployment_id: str) -> dict:
    """
    Refresh all guilds and channels for a Discord deployment.

    Fetches all guilds the bot has access to, then fetches all channels
    within each guild. Saves the results to MongoDB collections.

    Args:
        deployment_id: The deployment ID from deployments2 collection

    Returns:
        Dict with guilds_count, channels_count, and guilds list
    """
    import discord

    from eve.agent.session.models import ClientType, Deployment

    # Load deployment
    deployment = Deployment.from_mongo(ObjectId(deployment_id))
    if not deployment:
        raise ValueError(f"Deployment not found: {deployment_id}")
    if deployment.platform != ClientType.DISCORD:
        raise ValueError(f"Not a Discord deployment: {deployment_id}")
    if not deployment.secrets or not deployment.secrets.discord:
        raise ValueError(f"No Discord secrets found for deployment: {deployment_id}")

    token = deployment.secrets.discord.token

    # Create client and login
    client = discord.Client(intents=discord.Intents.default())
    await client.login(token)

    try:
        result = {"guilds_count": 0, "channels_count": 0, "guilds": []}

        deployment_oid = ObjectId(deployment_id)

        # Clear existing records for this deployment
        DiscordGuild.get_collection().delete_many({"deployment_id": deployment_oid})
        DiscordChannel.get_collection().delete_many({"deployment_id": deployment_oid})

        # Fetch all guilds
        async for guild in client.fetch_guilds():
            guild_data = {
                "deployment_id": deployment_oid,
                "guild_id": str(guild.id),
                "name": guild.name,
                "icon": str(guild.icon.url) if guild.icon else None,
                "member_count": None,  # Not available from fetch_guilds
                "channels": [],
            }

            # Fetch full guild to get channels
            try:
                full_guild = await client.fetch_guild(guild.id)
                # Update member count if available from full guild
                guild_data["member_count"] = getattr(
                    full_guild, "approximate_member_count", None
                )
                channels = await full_guild.fetch_channels()

                # Build category lookup
                categories = {c.id: c.name for c in channels if c.type.value == 4}

                for channel in channels:
                    channel_type_name = str(channel.type).replace("ChannelType.", "")
                    channel_data = {
                        "deployment_id": deployment_oid,
                        "guild_id": str(guild.id),
                        "channel_id": str(channel.id),
                        "name": channel.name,
                        "type": channel.type.value,
                        "type_name": channel_type_name,
                        "category_id": (
                            str(channel.category_id) if channel.category_id else None
                        ),
                        "category_name": categories.get(channel.category_id),
                        "position": channel.position,
                    }

                    # Save to collection
                    discord_channel = DiscordChannel(**channel_data)
                    discord_channel.save()

                    # Also embed in guild record
                    guild_data["channels"].append(
                        {
                            "id": str(channel.id),
                            "name": channel.name,
                            "type": channel.type.value,
                            "type_name": channel_type_name,
                            "category": channel_data["category_name"],
                        }
                    )

                    result["channels_count"] += 1

            except Exception as e:
                logger.warning(f"Failed to fetch channels for guild {guild.id}: {e}")

            # Save guild
            discord_guild = DiscordGuild(**guild_data)
            discord_guild.save()

            result["guilds"].append(
                {
                    "id": str(guild.id),
                    "name": guild.name,
                    "channel_count": len(guild_data["channels"]),
                }
            )
            result["guilds_count"] += 1

        logger.info(
            f"Refreshed Discord guilds/channels for deployment {deployment_id}: "
            f"{result['guilds_count']} guilds, {result['channels_count']} channels"
        )

        return result

    finally:
        await client.close()
