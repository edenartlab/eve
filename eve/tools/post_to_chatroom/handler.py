"""Handler for post_to_chatroom tool.

This tool allows agents in agent_sessions to post messages back to their
parent chatroom session. It's the primary way for agents in multi-agent
orchestration to contribute to the conversation after doing private work.
"""

from typing import List

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.models import ChatMessage, Session, prepare_result
from eve.tool import ToolContext
from eve.user import increment_message_count


def collect_urls_from_session_tool_results(
    session_id: ObjectId, limit: int = 20
) -> List[str]:
    """Collect URLs from recent tool results in a session.

    Looks at recent assistant messages with tool_calls and extracts URLs
    from their results. This helps auto-attach media the agent created.

    Args:
        session_id: The session to search for tool results
        limit: Max number of recent messages to check

    Returns:
        List of URLs found in tool results
    """
    urls = []

    # Get recent messages with tool calls
    messages = list(
        ChatMessage.get_collection().find(
            {
                "session": session_id,
                "role": "assistant",
                "tool_calls": {"$exists": True, "$ne": []},
            },
            sort=[("createdAt", -1)],
            limit=limit,
        )
    )

    for msg_doc in messages:
        tool_calls = msg_doc.get("tool_calls") or []
        for tc in tool_calls:
            result = prepare_result(tc.get("result"))
            if not result:
                continue
            # Extract URLs from tool result output
            for r in result:
                output = r.get("output", [])
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and item.get("url"):
                            urls.append(item["url"])
                elif isinstance(output, str) and output.startswith("http"):
                    # Some tools return URL directly as string
                    urls.append(output)

    return urls


async def handler(context: ToolContext):
    """Post a message from the agent_session to the parent chatroom.

    Args:
        context: ToolContext containing:
            - args.content: The message content to post
            - args.attachments: Optional list of media URLs to attach
            - agent: The agent ID posting the message
            - session: The agent_session ID (which has a parent_session)

    Returns:
        Dict with message_id, posted_to_session, and status

    Raises:
        Exception: If agent is not provided or session has no parent
    """
    if not context.agent:
        raise Exception("Agent is required")

    if not context.session:
        raise Exception("Session is required")

    # Get the current session (agent_session)
    agent_session = Session.from_mongo(context.session)
    if not agent_session:
        raise Exception(f"Agent session {context.session} not found")

    if not agent_session.parent_session:
        raise Exception(
            "This tool can only be used from an agent_session with a parent. "
            "The current session has no parent_session."
        )

    # Get the parent session (chatroom)
    parent_session = Session.from_mongo(agent_session.parent_session)
    if not parent_session:
        raise Exception(f"Parent session {agent_session.parent_session} not found")

    agent = Agent.from_mongo(context.agent)
    if not agent:
        raise Exception(f"Agent {context.agent} not found")

    content = context.args.get("content", "")
    attachments = context.args.get("attachments") or []

    # Auto-collect URLs from recent tool results if no attachments provided
    if not attachments:
        auto_urls = collect_urls_from_session_tool_results(agent_session.id)
        if auto_urls:
            attachments = auto_urls
            logger.info(
                f"[POST_TO_CHATROOM] Auto-collected {len(auto_urls)} URLs from tool results"
            )

    # Create message in the parent session
    new_message = ChatMessage(
        role="assistant",
        sender=agent.id,
        session=parent_session.id,
        content=content,
        attachments=attachments,
    )
    new_message.save()

    # Increment message count for the agent
    increment_message_count(agent.id)

    logger.info(
        f"[POST_TO_CHATROOM] Agent {agent.username} posted to parent session "
        f"{parent_session.id}: {content[:100]}..."
    )

    # Return success with message ID
    return {
        "output": [
            {
                "message_id": str(new_message.id),
                "posted_to_session": str(parent_session.id),
                "status": "success",
            }
        ]
    }
