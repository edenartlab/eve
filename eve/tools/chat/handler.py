"""Handler for chat tool.

This tool allows agents in agent_sessions to send messages - either public
(broadcast to all participants) or private (to specific recipients only).

Public messages (default): Posted to parent chatroom, distributed to all agent_sessions
Private messages: Only stored in parent session (for audit) and recipients' workspaces
"""

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.models import (
    Channel,
    ChatMessage,
    EdenMessageAgentData,
    EdenMessageData,
    EdenMessageType,
    Session,
)
from eve.tool import ToolContext
from eve.user import increment_message_count


async def handler(context: ToolContext):
    """Send a message from the agent_session.

    Supports two modes:
    - Public (default): Message posted to parent chatroom, visible to all
    - Private: Message only visible to specified recipients

    Args:
        context: ToolContext containing:
            - args.content: The message content to send
            - args.attachments: Optional list of media URLs to attach
            - args.public: Boolean (default True) - public or private message
            - args.recipients: List of agent usernames (required if public=false)
            - agent: The agent ID sending the message
            - session: The agent_session ID (which has a parent_session)

    Returns:
        Dict with message_id, type, and status

    Raises:
        Exception: If validation fails or required data is missing
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

    sender = Agent.from_mongo(context.agent)
    if not sender:
        raise Exception(f"Agent {context.agent} not found")

    content = context.args.get("content", "")
    attachments = context.args.get("attachments") or []
    public = context.args.get("public", True)
    recipients = context.args.get("recipients") or []

    # Upload attachments to eden if present
    if attachments:
        from eve.s3 import upload_attachments_to_eden

        attachments = await upload_attachments_to_eden(attachments)

    # === VALIDATION ===
    if public:
        # Public mode: recipients must NOT be set
        if recipients:
            raise Exception(
                "Cannot specify recipients for a public message. "
                "Set public=false to send a private message."
            )
        return await _handle_public_message(
            sender, parent_session, content, attachments
        )
    else:
        # Private mode: recipients required
        if not recipients:
            raise Exception(
                "Recipients are required for private messages. "
                "Provide a list of agent usernames."
            )
        return await _handle_private_message(
            sender, parent_session, content, attachments, recipients
        )


async def _handle_public_message(
    sender: Agent,
    parent_session: Session,
    content: str,
    attachments: list,
) -> dict:
    """Handle public broadcast message (existing behavior)."""

    # Create message in the parent session
    new_message = ChatMessage(
        role="assistant",
        sender=sender.id,
        session=[parent_session.id],
        content=content,
        attachments=attachments,
    )
    new_message.save()

    # Add channel reference for tracking
    channel = Channel(type="eden", key=str(new_message.id))
    new_message.update(channel=channel.model_dump())

    # Distribute to OTHER agent_sessions (exclude the posting agent's own session)
    if parent_session.agent_sessions and len(parent_session.agent_sessions) > 0:
        from eve.agent.session.context import distribute_message_to_agent_sessions

        await distribute_message_to_agent_sessions(
            parent_session=parent_session,
            message=new_message,
            exclude_agent_id=sender.id,  # Don't send to the agent who just posted
        )

    # Increment message count for the agent
    increment_message_count(sender.id)

    logger.info(
        f"[CHAT] Public: {sender.username} posted to session "
        f"{parent_session.id}: {content[:100]}..."
    )

    return {
        "output": [
            {
                "message_id": str(new_message.id),
                "posted_to_session": str(parent_session.id),
                "type": "public",
                "status": "success",
            }
        ]
    }


async def _handle_private_message(
    sender: Agent,
    parent_session: Session,
    content: str,
    attachments: list,
    recipient_usernames: list,
) -> dict:
    """Handle private message to specific recipients."""

    if not parent_session.agent_sessions:
        raise Exception("No agent_sessions configured for this session")

    # Build username -> agent mapping from session agents
    agents_in_session = {}
    for agent_id_str in parent_session.agent_sessions.keys():
        agent = Agent.from_mongo(ObjectId(agent_id_str))
        if agent:
            agents_in_session[agent.username.lower()] = agent

    # Validate recipients
    validated_recipients = []
    invalid_usernames = []

    for username in recipient_usernames:
        username_lower = username.lower()

        # Check for self-messaging
        if username_lower == sender.username.lower():
            raise Exception(
                f"Cannot send a private message to yourself ({sender.username})"
            )

        # Check if valid agent in session
        if username_lower in agents_in_session:
            validated_recipients.append(agents_in_session[username_lower])
        else:
            invalid_usernames.append(username)

    if invalid_usernames:
        valid_names = [
            a.username for a in agents_in_session.values() if a.id != sender.id
        ]
        raise Exception(
            f"Invalid recipient(s): {invalid_usernames}. "
            f"Valid recipients: {valid_names}"
        )

    if not validated_recipients:
        raise Exception("No valid recipients specified")

    # Build recipient list for logging and response
    recipient_names = [r.username for r in validated_recipients]
    recipient_list_str = ", ".join(recipient_names)

    # Create eden message data with sender and recipient info
    eden_data = EdenMessageData(
        message_type=EdenMessageType.PRIVATE_MESSAGE,
        agents=[
            EdenMessageAgentData(
                id=r.id,
                name=r.name or r.username,
                avatar=r.userImage,
            )
            for r in validated_recipients
        ],
        sender=EdenMessageAgentData(
            id=sender.id,
            name=sender.name or sender.username,
            avatar=sender.userImage,
        ),
    )

    # Determine target session IDs:
    # - Parent session (for storage/audit, will be filtered from conductor view)
    # - Each recipient's agent_session (they will see the message)
    # Note: Sender's workspace is NOT included - they have the tool call args already
    target_session_ids = [parent_session.id]

    for recipient in validated_recipients:
        recipient_session_id = parent_session.agent_sessions.get(str(recipient.id))
        if recipient_session_id:
            target_session_ids.append(recipient_session_id)

    # Create the private message with eden role
    new_message = ChatMessage(
        role="eden",  # System-style message for private DMs
        sender=sender.id,
        session=target_session_ids,
        content=content,
        attachments=attachments,
        eden_message_data=eden_data,
    )
    new_message.save()

    # Add channel reference for tracking
    channel = Channel(type="eden", key=str(new_message.id))
    new_message.update(channel=channel.model_dump())

    # Increment message count for the sender
    increment_message_count(sender.id)

    logger.info(
        f"[CHAT] Private: {sender.username} -> {recipient_list_str} "
        f"(sessions: {[str(s) for s in target_session_ids]}): {content[:100]}..."
    )

    return {
        "output": [
            {
                "message_id": str(new_message.id),
                "recipients": recipient_names,
                "type": "private",
                "status": "success",
            }
        ]
    }
