"""Handler for post_to_chatroom tool.

This tool allows agents in agent_sessions to post messages back to their
parent chatroom session. It's the primary way for agents in multi-agent
orchestration to contribute to the conversation after doing private work.
"""

from loguru import logger

from eve.agent import Agent
from eve.agent.session.models import Channel, ChatMessage, Session
from eve.tool import ToolContext
from eve.user import increment_message_count


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
    if attachments:
        from eve.s3 import upload_attachments_to_eden

        attachments = await upload_attachments_to_eden(attachments)

    # Create message in the parent session
    new_message = ChatMessage(
        role="assistant",
        sender=agent.id,
        session=[parent_session.id],
        content=content,
        attachments=attachments,
    )
    new_message.save()

    # Add channel reference for tracking (use update() to only modify channel field)
    # Convert to dict for MongoDB encoding
    channel = Channel(type="eden", key=str(new_message.id))
    new_message.update(channel=channel.model_dump())

    # Distribute to OTHER agent_sessions (exclude the posting agent's own session)
    if parent_session.agent_sessions and len(parent_session.agent_sessions) > 0:
        from eve.agent.session.context import distribute_message_to_agent_sessions

        await distribute_message_to_agent_sessions(
            parent_session=parent_session,
            message=new_message,
            exclude_agent_id=agent.id,  # Don't send to the agent who just posted
        )

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
