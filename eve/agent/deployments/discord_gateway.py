"""
Discord Gateway Helper Models and Functions

Provides MongoDB models for tracking Discord messages and helper functions
for message backfilling and media handling.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from bson import ObjectId
from loguru import logger

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
    session_id: Optional[ObjectId] = None  # Eve session ID
    eve_message_id: Optional[ObjectId] = None  # Eve ChatMessage ID
    processed: bool = False
    first_seen_at: datetime = None
    last_seen_at: datetime = None

    def __init__(self, **data):
        if "first_seen_at" not in data:
            data["first_seen_at"] = datetime.now(timezone.utc)
        if "last_seen_at" not in data:
            data["last_seen_at"] = datetime.now(timezone.utc)
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
    max_messages: int = 100,
    max_days: int = 7,
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
            s3_url = await upload_file_from_url(url)
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
) -> DiscordMessage:
    """
    Get existing DiscordMessage or create a new one.

    Args:
        message_data: Raw Discord message data
        session_id: Eve session ID to link
        eve_message_id: Eve ChatMessage ID to link

    Returns:
        DiscordMessage document
    """
    discord_msg_id = message_data.get("id")

    # Try to find existing by Discord message ID
    existing = DiscordMessage.find_one({"discord_message_id": discord_msg_id})
    if existing:
        # Update linkage if provided
        updates = {"last_seen_at": datetime.now(timezone.utc)}
        if session_id:
            updates["session_id"] = session_id
        if eve_message_id:
            updates["eve_message_id"] = eve_message_id
        existing.update(**updates)
        return existing

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
        session_id=session_id,
        eve_message_id=eve_message_id,
    )
    discord_msg.save()
    return discord_msg
