"""Agent session runtime for multi-agent orchestration.

Handles the private workspace flow:
1. Sync new parent messages to agent_session
2. Run LLM + tool processing until post_to_chatroom
3. Update sync point

This module bridges the gap between the parent chatroom session and each
agent's private workspace (agent_session).
"""

import uuid

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.context import (
    build_agent_session_llm_context,
    format_parent_messages_for_agent_session,
    get_new_parent_messages,
)
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    PromptSessionContext,
    Session,
    UpdateType,
)
from eve.agent.session.runtime import PromptSessionRuntime


async def run_agent_session_turn(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
) -> None:
    """Run a single turn in an agent_session.

    This function:
    1. Loads the agent_session
    2. Gets new messages from the parent since last sync
    3. Saves the bulk update as a user message in agent_session (for history)
    4. Builds LLM context with chatroom framing
    5. Runs the prompt loop until the agent posts to chatroom
    6. Updates the sync point (last_parent_message_id)

    The agent works privately until they call post_to_chatroom,
    which posts to the parent session.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session
    """
    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(
        f"[AGENT_SESSION] Running turn for {actor.username} in session {agent_session_id}"
    )

    # Generate session run ID for this turn
    session_run_id = str(uuid.uuid4())

    # Build context for this turn
    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Get new messages from parent since last sync
    new_parent_messages = get_new_parent_messages(
        parent_session,
        agent_session.last_parent_message_id,
    )

    # Save the bulk update as a user message in agent_session for history
    # This ensures the agent can see context in future turns
    if new_parent_messages:
        bulk_content = format_parent_messages_for_agent_session(
            new_parent_messages, actor.id
        )
        sync_message = ChatMessage(
            session=agent_session.id,
            role="user",
            sender=ObjectId("000000000000000000000000"),  # System sender
            content=bulk_content,
        )
        sync_message.save()
        logger.info(
            f"[AGENT_SESSION] Saved {len(new_parent_messages)} parent messages as bulk update"
        )

    # Build LLM context with parent message sync
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
    )

    # Run the prompt loop using the standard runtime
    runtime = PromptSessionRuntime(
        session=agent_session,
        llm_context=llm_context,
        actor=actor,
        stream=False,
        is_client_platform=False,
        session_run_id=session_run_id,
        api_key_id=None,
        context=context,
    )

    posted_to_parent = False
    async for update in runtime.run():
        logger.debug(f"[AGENT_SESSION] Update: {update.type}")

        # Check if agent posted to parent via tool
        if (
            update.type == UpdateType.TOOL_COMPLETE
            and update.tool_name == "post_to_chatroom"
        ):
            posted_to_parent = True
            logger.info(f"[AGENT_SESSION] {actor.username} posted to chatroom")

    # Update last_parent_message_id to track sync point
    if new_parent_messages:
        last_msg_id = new_parent_messages[-1].id
        agent_session.update(last_parent_message_id=last_msg_id)
        logger.info(f"[AGENT_SESSION] Updated sync point to message {last_msg_id}")

    if not posted_to_parent:
        logger.warning(
            f"[AGENT_SESSION] {actor.username} completed turn without posting to chatroom"
        )


async def run_agent_session_turn_streaming(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
):
    """Streaming version of run_agent_session_turn.

    Yields SessionUpdates as they occur, allowing real-time streaming
    of the agent's work.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session

    Yields:
        SessionUpdate objects as the agent works
    """
    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(
        f"[AGENT_SESSION] Running streaming turn for {actor.username} in {agent_session_id}"
    )

    session_run_id = str(uuid.uuid4())

    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Get and save new parent messages
    new_parent_messages = get_new_parent_messages(
        parent_session,
        agent_session.last_parent_message_id,
    )

    if new_parent_messages:
        bulk_content = format_parent_messages_for_agent_session(
            new_parent_messages, actor.id
        )
        sync_message = ChatMessage(
            session=agent_session.id,
            role="user",
            sender=ObjectId("000000000000000000000000"),
            content=bulk_content,
        )
        sync_message.save()

    # Build LLM context
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
    )

    # Run with streaming
    runtime = PromptSessionRuntime(
        session=agent_session,
        llm_context=llm_context,
        actor=actor,
        stream=True,  # Enable streaming
        is_client_platform=False,
        session_run_id=session_run_id,
        api_key_id=None,
        context=context,
    )

    posted_to_parent = False
    async for update in runtime.run():
        yield update

        if (
            update.type == UpdateType.TOOL_COMPLETE
            and update.tool_name == "post_to_chatroom"
        ):
            posted_to_parent = True

    # Update sync point
    if new_parent_messages:
        last_msg_id = new_parent_messages[-1].id
        agent_session.update(last_parent_message_id=last_msg_id)

    if not posted_to_parent:
        logger.warning(
            f"[AGENT_SESSION] {actor.username} completed streaming turn without posting"
        )
