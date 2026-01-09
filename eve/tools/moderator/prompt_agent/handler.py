"""Handler for moderator prompt_agent tool.

This tool prompts a specific agent to take their turn in the conversation.
It is SYNCHRONOUS - it waits for the agent to complete before returning.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.automatic import get_reply_delay
from eve.agent.session.models import (
    ChatMessage,
    EdenMessageData,
    EdenMessageType,
    Session,
)
from eve.tool import ToolContext


async def handler(context: ToolContext) -> Dict[str, Any]:
    """Prompt an agent to take their turn.

    SYNCHRONOUS: Waits for the agent to complete their turn.

    Args:
        context: ToolContext containing:
            - args.agent_username: Username of agent to prompt
            - args.hint: Optional factual state update
            - session: The moderator_session ID (which has a parent_session)

    Returns:
        Dict with agent's response content

    Raises:
        Exception: If validation fails or required data is missing
    """
    if not context.session:
        raise Exception("Session is required")

    # Get the moderator session
    moderator_session = Session.from_mongo(context.session)
    if not moderator_session:
        raise Exception(f"Moderator session {context.session} not found")

    if not moderator_session.parent_session:
        raise Exception(
            "This tool can only be used from a moderator_session with a parent. "
            "The current session has no parent_session."
        )

    # Get the parent session
    parent_session = Session.from_mongo(moderator_session.parent_session)
    if not parent_session:
        raise Exception(f"Parent session {moderator_session.parent_session} not found")

    # Check agent_sessions exist
    if not parent_session.agent_sessions:
        raise Exception(
            "No agent_sessions found. Call start_session first to initialize agents."
        )

    # Parse args
    agent_username = context.args.get("agent_username", "")
    hint = context.args.get("hint")

    if not agent_username:
        raise Exception("agent_username is required")

    # Find the agent by username
    target_agent = None
    target_agent_session_id = None

    for agent_id_str, agent_session_id in parent_session.agent_sessions.items():
        agent = Agent.from_mongo(ObjectId(agent_id_str))
        if agent and agent.username.lower() == agent_username.lower():
            target_agent = agent
            target_agent_session_id = agent_session_id
            break

    if not target_agent:
        # List valid agent usernames for error message
        valid_usernames = []
        for agent_id_str in parent_session.agent_sessions.keys():
            agent = Agent.from_mongo(ObjectId(agent_id_str))
            if agent:
                valid_usernames.append(agent.username)
        raise Exception(
            f"Agent '{agent_username}' not found in session. "
            f"Valid agents: {valid_usernames}"
        )

    logger.info(
        f"[MODERATOR_PROMPT] Prompting {target_agent.username} in session {parent_session.id}"
    )

    # Send hint to agent's workspace (if provided)
    if hint:
        hint_content = f"""🎯 IT'S YOUR TURN!

{hint}

⚠️ IMPORTANT: You MUST use the chat tool to respond. Your response will not be seen by others unless you post it."""
    else:
        hint_content = """🎯 IT'S YOUR TURN!

⚠️ IMPORTANT: You MUST use the chat tool to respond. Your response will not be seen by others unless you post it."""

    hint_message = ChatMessage(
        session=[target_agent_session_id],  # Only to this agent's workspace
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content=hint_content,
        eden_message_data=EdenMessageData(
            message_type=EdenMessageType.MODERATOR_PROMPT
        ),
    )
    hint_message.save()
    logger.info(f"[MODERATOR_PROMPT] Sent hint to {target_agent.username}'s workspace")

    # Record the prompt action in the parent session
    prompt_record = ChatMessage(
        session=[parent_session.id],
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content=f'{{"agent": "{target_agent.username}", "hint": {repr(hint) if hint else "null"}}}',
        eden_message_data=EdenMessageData(
            message_type=EdenMessageType.MODERATOR_PROMPT
        ),
    )
    prompt_record.save()

    # Run the agent's turn synchronously
    from eve.agent.session.agent_session_runtime import run_agent_session_turn

    await run_agent_session_turn(
        parent_session=parent_session,
        agent_session_id=target_agent_session_id,
        actor=target_agent,
    )

    logger.info(
        f"[MODERATOR_PROMPT] Agent {target_agent.username} completed their turn"
    )

    # Apply delay_interval between agent turns (for automatic sessions)
    # Reload parent session to check if still active
    parent_session.reload()
    if parent_session.session_type == "automatic" and parent_session.status not in (
        "finished",
        "paused",
        "archived",
    ):
        delay = get_reply_delay(parent_session)
        if delay > 0:
            # Set waiting_until so client knows we're in a delay period
            wait_until = datetime.now(timezone.utc) + timedelta(seconds=delay)
            parent_session.update(waiting_until=wait_until)
            logger.info(
                f"[MODERATOR_PROMPT] >>>>>> WAITING {delay} SECONDS BEFORE NEXT AGENT (until {wait_until.isoformat()}) <<<<<<"
            )
            await asyncio.sleep(delay)
            logger.info(f"[MODERATOR_PROMPT] >>>>>> DELAY COMPLETE ({delay}s) <<<<<<")
            # Clear waiting_until
            parent_session.update(waiting_until=None)

    # Find the agent's response (the message they posted via chat)
    # This is the most recent assistant message from this agent in the parent session
    recent_messages = list(
        ChatMessage.get_collection()
        .find(
            {
                "session": parent_session.id,
                "sender": target_agent.id,
                "role": "assistant",
            }
        )
        .sort("createdAt", -1)
        .limit(1)
    )

    agent_response = ""
    if recent_messages:
        agent_response = recent_messages[0].get("content", "")

    return {
        "output": {
            "status": "success",
            "agent": target_agent.username,
            "response": agent_response,
        }
    }
