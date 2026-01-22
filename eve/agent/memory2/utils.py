"""
Memory System v2 - Shared Utilities

This module contains utility functions that are shared across the memory system
and other parts of the codebase (session, handlers, etc.).
"""

import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT
from eve.agent.session.models import ChatMessage, Session
from eve.user import User

# Multipliers on the char/token count of different message types for memory importance / formation triggers:
USER_MULTIPLIER = 1.0
TOOL_MULTIPLIER = 0.5  # mostly discord_search rn
AGENT_MULTIPLIER = 0.2  # we want memories to come from users, not agents
OTHER_MULTIPLIER = 0.5  # not really used


def get_sender_id_to_sender_name_map(
    messages: List[ChatMessage],
) -> Dict[ObjectId, str]:
    """Find all unique senders in the messages and return a map of sender id to sender name"""
    unique_sender_ids = {msg.sender for msg in messages if msg.sender}

    if not unique_sender_ids:
        return {}

    # Perform single MongoDB query with projection to fetch only needed fields
    try:
        from eve.mongo import get_collection

        users_collection = get_collection(User.collection_name)

        # Query with projection to only get _id, username, and type fields
        users_cursor = users_collection.find(
            {"_id": {"$in": list(unique_sender_ids)}},
            {"_id": 1, "username": 1, "type": 1},
        )

        sender_id_to_sender_name_map = {}

        # Process each user from cursor safely
        for user in users_cursor:
            try:
                sender_id_to_sender_name_map[user["_id"]] = (
                    f"{user['username']} ({user['type']})"
                )
            except (KeyError, TypeError) as e:
                logger.error(f"Error processing user {user.get('_id', 'unknown')}: {e}")
                traceback.print_exc()
                if "_id" in user:
                    sender_id_to_sender_name_map[user["_id"]] = "unknown"

        # Ensure all unique_sender_ids are covered, defaulting to "unknown" for missing ones
        for sender_id in unique_sender_ids:
            if sender_id not in sender_id_to_sender_name_map:
                sender_id_to_sender_name_map[sender_id] = "unknown"

        return sender_id_to_sender_name_map

    except Exception as e:
        logger.error(f"Error in get_sender_id_to_sender_name_map(): {e}")
        traceback.print_exc()
        return {}


def messages_to_text(
    messages: List[ChatMessage], skip_trigger_messages: bool = True
) -> tuple[str, dict[str, int]]:
    """Convert messages to readable text for LLM processing

    Args:
        messages: List of ChatMessage objects to convert
        skip_trigger_messages: If True, skip messages that originated from triggers (default: True)

    Returns:
        tuple: (formatted_text, char_counts_by_source)
        - formatted_text: The fully formatted message string
        - char_counts_by_source: Dictionary with character counts per message type
    """
    sender_id_to_sender_name_map = get_sender_id_to_sender_name_map(messages)
    text_parts = []
    char_counts_by_source = {"user": 0, "agent": 0, "tool": 0, "other": 0}

    for msg in messages:
        # Skip system messages (e.g., periodic task instructions)
        if msg.role == "system":
            continue

        # Skip trigger messages if requested
        if skip_trigger_messages and msg.trigger:
            continue
        speaker = sender_id_to_sender_name_map.get(msg.sender) or msg.name or msg.role
        content = msg.content

        # Count original content characters by message type
        if msg.role == "user":
            char_counts_by_source["user"] += len(content)
        elif msg.role in ["agent", "assistant", "eden"]:
            char_counts_by_source["agent"] += len(content)
        else:
            char_counts_by_source["other"] += len(content)

        if msg.tool_calls:  # Add tool calls summary if present
            tools_summary = (
                f" [Used tools: {', '.join([tc.tool for tc in msg.tool_calls])}]"
            )
            content += tools_summary
            char_counts_by_source["tool"] += len(tools_summary)

            # Include full tool call results for specific tools
            tools_with_full_results = [
                "discord_search"
            ]  # , "farcaster_search", "twitter_search", "get_tweets"]
            for tc in msg.tool_calls:
                if (
                    tc.result
                    and tc.status == "completed"
                    and tc.tool in tools_with_full_results
                ):
                    try:
                        import json

                        result_json = json.dumps(tc.result)
                        tool_result_content = (
                            f"\n[{tc.tool} full result: {result_json}]"
                        )
                        content += tool_result_content
                        char_counts_by_source["tool"] += len(tool_result_content)
                    except Exception:
                        # Fallback: just mention tool was used with results
                        fallback_content = f"\n[{tc.tool} completed with results]"
                        content += fallback_content
                        char_counts_by_source["tool"] += len(fallback_content)
        text_parts.append(f"{speaker}: {content}")
    return "\n\n".join(text_parts), char_counts_by_source


def select_messages(
    session: Session,
    selection_limit: Optional[int] = DEFAULT_SESSION_SELECTION_LIMIT,
    last_memory_message_id: Optional[ObjectId] = None,
):
    """
    Select messages from a session with an optional selection limit.

    Args:
        session: The session to select messages from
        selection_limit: Maximum number of recent messages to select
        last_memory_message_id: If provided, ensures we fetch enough messages to include
                                everything since this message (takes max of this and selection_limit)
    """
    messages = ChatMessage.get_collection()

    # If last_memory_message_id is provided, calculate how many messages we need
    effective_limit = selection_limit
    if last_memory_message_id is not None and selection_limit is not None:
        # Get the timestamp of the last memory message
        last_memory_msg = messages.find_one(
            {"_id": last_memory_message_id}, {"createdAt": 1}
        )

        if last_memory_msg and last_memory_msg.get("createdAt"):
            # Count messages since last_memory_message_id (including messages created after it)
            messages_since_last = messages.count_documents(
                {
                    "session": session.id,
                    "createdAt": {"$gt": last_memory_msg["createdAt"]},
                }
            )

            # Use the larger of selection_limit or messages_since_last to ensure we get everything
            # Add a small buffer to account for edge cases
            effective_limit = max(selection_limit, messages_since_last + 5)

    # Select all messages including eden messages - they all share the same limit
    selected_messages = messages.find({"session": session.id}).sort("createdAt", -1)

    if effective_limit is not None:
        selected_messages = selected_messages.limit(effective_limit)
    selected_messages = list(selected_messages)

    pinned_messages = messages.find({"session": session.id, "pinned": True})
    pinned_messages = list(pinned_messages)
    pinned_messages = [
        m
        for m in pinned_messages
        if m["_id"] not in [msg["_id"] for msg in selected_messages]
    ]
    selected_messages.extend(pinned_messages)

    selected_messages.reverse()

    # Convert to ChatMessage objects, skipping any with validation errors
    valid_messages = []
    for msg in selected_messages:
        try:
            valid_messages.append(ChatMessage(**msg))
        except Exception as e:
            logger.warning(f"Skipping message {msg.get('_id')} in memory: {e}")

    # Filter out cancelled tool calls from the messages
    valid_messages = [msg.filter_cancelled_tool_calls() for msg in valid_messages]

    return valid_messages


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4.5 characters per token)"""
    return int(len(text) / 4.5)


def get_agent_owner(agent_id: ObjectId) -> Optional[ObjectId]:
    """Get the owner of the agent"""
    try:
        from eve.agent.agent import Agent

        agent = Agent.from_mongo(agent_id)
        return agent.owner
    except Exception as e:
        logger.error(f"Warning: Could not load agent owner for {agent_id}: {e}")
        traceback.print_exc()
        return None
