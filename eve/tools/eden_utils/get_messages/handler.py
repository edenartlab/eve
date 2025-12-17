import json
from datetime import datetime, timedelta, timezone
from typing import List

from bson import ObjectId
from jinja2 import Template
from loguru import logger

from eve.agent.session.models import ChatMessage, Session
from eve.tool import ToolContext
from eve.user import User
from eve.utils import prepare_result

# Tools that should include args in the output
TOOLS_WITH_ARGS = {"tweet", "discord_post", "farcaster_cast", "post_to_chatroom"}

message_template = Template("""[{{timestamp}}] {{username}}: {{content}}""")

session_template = Template(
    """<Session id="{{session_id}}"{% if title %} title="{{title}}"{% endif %}>
{{messages}}
</Session>"""
)


def get_username(sender_id: ObjectId) -> str:
    """Get username from sender ObjectId, with fallback."""
    if not sender_id:
        return "unknown"
    user = User.from_mongo(sender_id)
    if user and user.username:
        return user.username
    return "unknown"


def extract_urls_from_data(data) -> List[str]:
    """Recursively extract all URL strings from any data structure."""
    urls = []
    if isinstance(data, str):
        return urls
    if isinstance(data, dict):
        # If this dict has a url key, extract it
        if "url" in data and isinstance(data["url"], str):
            urls.append(data["url"])
        # Also recurse into all values
        for v in data.values():
            urls.extend(extract_urls_from_data(v))
    elif isinstance(data, list):
        for item in data:
            urls.extend(extract_urls_from_data(item))
    return urls


def extract_urls_from_result(result) -> List[str]:
    """Extract media URLs from tool call result as strings only."""
    if not result:
        return []
    prepared = prepare_result(result)
    if not prepared:
        return []
    return extract_urls_from_data(prepared)


def format_result_summary(result) -> str:
    """Format tool result as a brief summary with actual URLs."""
    if not result:
        return ""
    prepared = prepare_result(result)
    if not prepared:
        return ""

    # Extract all URLs from the result
    urls = extract_urls_from_data(prepared)
    return " ".join(urls) if urls else ""


def format_tool_call(tc) -> str:
    """Format a tool call with optional args for specific tools."""
    result_summary = format_result_summary(tc.result)

    # Include args for specific posting tools
    if tc.tool in TOOLS_WITH_ARGS:
        args_str = json.dumps(tc.args, ensure_ascii=False)
        if result_summary:
            return f"[Tool: {tc.tool} args={args_str} -> {result_summary}]"
        else:
            return f"[Tool: {tc.tool} args={args_str}]"
    else:
        if result_summary:
            return f"[Tool: {tc.tool} -> {result_summary}]"
        else:
            return f"[Tool: {tc.tool}]"


async def handler(context: ToolContext):
    session_ids = context.args.get("session_ids", [])
    hours = context.args.get("hours", 24)
    agent_id = context.agent

    # Calculate the cutoff time
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Convert string IDs to ObjectIds and filter by agent membership if agent_id provided
    object_ids = []
    if agent_id:
        if isinstance(agent_id, str):
            agent_id = ObjectId(agent_id)

    for sid in session_ids:
        session_oid = ObjectId(sid) if isinstance(sid, str) else sid

        # If agent_id is provided, filter to sessions where agent is a participant
        if agent_id:
            session = Session.find_one({"_id": session_oid})
            if session and agent_id in session.agents:
                object_ids.append(session_oid)
            else:
                logger.warning(
                    f"Filtering out session {sid}: agent {agent_id} is not in session.agents"
                )
        else:
            object_ids.append(session_oid)

    if not object_ids:
        return {
            "output": {"messages": "No valid session IDs provided.", "attachments": []}
        }

    # Build output for each session
    session_outputs = []
    all_attachments = []

    for session_id in object_ids:
        # Get session info for title
        session = Session.find_one({"_id": session_id})
        session_title = session.title if session else None

        # Query messages for this session within the time window
        # Exclude messages with role "eden"
        messages = ChatMessage.find(
            {
                "session": session_id,
                "createdAt": {"$gte": cutoff_time},
                "role": {"$ne": "eden"},
            }
        )

        # Sort by creation time
        messages = sorted(messages, key=lambda x: x.createdAt)

        if not messages:
            session_outputs.append(
                session_template.render(
                    session_id=str(session_id),
                    title=session_title,
                    messages="(No messages in the specified time window)",
                )
            )
            continue

        # Format each message
        formatted_messages = []
        for msg in messages:
            # Format timestamp as HH:MM without seconds or UTC
            timestamp = msg.createdAt.strftime("%Y-%m-%d %H:%M")
            content = msg.content.strip() if msg.content else ""

            # Get username from sender
            username = get_username(msg.sender)

            # Collect message attachments (extract URL if dict, otherwise use as-is)
            message_attachments = []
            if msg.attachments:
                for att in msg.attachments:
                    if isinstance(att, dict) and att.get("url"):
                        message_attachments.append(att["url"])
                        all_attachments.append(att["url"])
                    elif isinstance(att, str):
                        message_attachments.append(att)
                        all_attachments.append(att)

            # Include tool call info with results if present
            if msg.tool_calls:
                tool_info = []
                for tc in msg.tool_calls:
                    tool_info.append(format_tool_call(tc))

                    # Extract media URLs from tool results
                    result_urls = extract_urls_from_result(tc.result)
                    all_attachments.extend(result_urls)

                if tool_info:
                    content = (
                        f"{content}\n{' '.join(tool_info)}"
                        if content
                        else " ".join(tool_info)
                    )

            # Add channel URL if present (permanent link to message on platform)
            if msg.channel and msg.channel.url:
                content = (
                    f"{content} ({msg.channel.url})"
                    if content
                    else f"({msg.channel.url})"
                )

            # Add message attachments to content
            if message_attachments:
                attachments_str = ", ".join(message_attachments)
                content = (
                    f"{content} [attachments: {attachments_str}]"
                    if content
                    else f"[attachments: {attachments_str}]"
                )

            formatted_messages.append(
                message_template.render(
                    timestamp=timestamp,
                    username=username,
                    content=content,
                )
            )

        session_outputs.append(
            session_template.render(
                session_id=str(session_id),
                title=session_title,
                messages="\n".join(formatted_messages),
            )
        )

    # Combine all sessions into a single output string
    messages_output = "\n\n".join(session_outputs)

    return {"output": {"messages": messages_output, "attachments": all_attachments}}
