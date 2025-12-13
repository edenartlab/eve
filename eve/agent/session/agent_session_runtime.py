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

    Note: This uses PromptSessionRuntime directly (not orchestrator) because
    it requires specialized context building via build_agent_session_llm_context.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session
    """
    logger.info("[AGENT_SESSION] ========== run_agent_session_turn START ==========")
    logger.info(f"[AGENT_SESSION] Parent session: {parent_session.id}")
    logger.info(f"[AGENT_SESSION] Agent session ID: {agent_session_id}")
    logger.info(f"[AGENT_SESSION] Actor: {actor.username} (id={actor.id})")

    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        logger.error(f"[AGENT_SESSION] Agent session {agent_session_id} not found")
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(f"[AGENT_SESSION] Loaded agent_session: title='{agent_session.title}'")
    logger.info(
        f"[AGENT_SESSION] Last parent message ID: {agent_session.last_parent_message_id}"
    )

    # Generate session run ID for this turn
    session_run_id = str(uuid.uuid4())
    logger.info(f"[AGENT_SESSION] Session run ID: {session_run_id}")

    # Build context for this turn
    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Get new messages from parent since last sync
    logger.info("[AGENT_SESSION] ===== Parent Message Sync =====")
    new_parent_messages = get_new_parent_messages(
        parent_session,
        agent_session.last_parent_message_id,
    )
    logger.info(
        f"[AGENT_SESSION] Found {len(new_parent_messages)} new parent messages to sync"
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
            f"[AGENT_SESSION] Saved bulk update message (first 200 chars): {bulk_content[:200]}..."
        )
    else:
        logger.info("[AGENT_SESSION] No new parent messages to sync")

    # Build LLM context with parent message sync
    logger.info("[AGENT_SESSION] ===== Building LLM Context =====")
    logger.info("[AGENT_SESSION] Calling build_agent_session_llm_context...")
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
    )
    logger.info("[AGENT_SESSION] LLM context built successfully")

    # Run the prompt loop using the standard runtime
    # Note: Using PromptSessionRuntime directly for specialized agent session handling
    logger.info("[AGENT_SESSION] ===== Running PromptSessionRuntime =====")
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
    update_count = 0
    logger.info("[AGENT_SESSION] Starting runtime.run() loop...")

    async for update in runtime.run():
        update_count += 1
        logger.info(f"[AGENT_SESSION] Update #{update_count}: type={update.type}")

        # Check if agent posted to parent via tool
        if (
            update.type == UpdateType.TOOL_COMPLETE
            and update.tool_name == "post_to_chatroom"
        ):
            posted_to_parent = True
            logger.info(f"[AGENT_SESSION] >>> {actor.username} posted to chatroom <<<")

    logger.info(f"[AGENT_SESSION] Runtime loop completed, {update_count} updates")

    # Update last_parent_message_id to track sync point
    logger.info("[AGENT_SESSION] ===== Updating Sync Point =====")
    if new_parent_messages:
        last_msg_id = new_parent_messages[-1].id
        agent_session.update(last_parent_message_id=last_msg_id)
        logger.info(f"[AGENT_SESSION] Updated sync point to message {last_msg_id}")
    else:
        logger.info("[AGENT_SESSION] No sync point update needed")

    if posted_to_parent:
        logger.info(
            f"[AGENT_SESSION] Turn completed: {actor.username} posted to chatroom"
        )
    else:
        logger.warning(
            f"[AGENT_SESSION] Turn completed: {actor.username} did NOT post to chatroom"
        )

    logger.info("[AGENT_SESSION] ========== run_agent_session_turn END ==========")


async def run_agent_session_turn_streaming(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
):
    """Streaming version of run_agent_session_turn.

    Yields SessionUpdates as they occur, allowing real-time streaming
    of the agent's work.

    Note: This uses PromptSessionRuntime directly (not orchestrator) because
    it requires specialized context building via build_agent_session_llm_context.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session

    Yields:
        SessionUpdate objects as the agent works
    """
    logger.info(
        "[AGENT_SESSION_STREAM] ========== run_agent_session_turn_streaming START =========="
    )
    logger.info(f"[AGENT_SESSION_STREAM] Parent session: {parent_session.id}")
    logger.info(f"[AGENT_SESSION_STREAM] Agent session ID: {agent_session_id}")
    logger.info(f"[AGENT_SESSION_STREAM] Actor: {actor.username} (id={actor.id})")

    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        logger.error(
            f"[AGENT_SESSION_STREAM] Agent session {agent_session_id} not found"
        )
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(
        f"[AGENT_SESSION_STREAM] Loaded agent_session: title='{agent_session.title}'"
    )

    session_run_id = str(uuid.uuid4())
    logger.info(f"[AGENT_SESSION_STREAM] Session run ID: {session_run_id}")

    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Get and save new parent messages
    logger.info("[AGENT_SESSION_STREAM] ===== Parent Message Sync =====")
    new_parent_messages = get_new_parent_messages(
        parent_session,
        agent_session.last_parent_message_id,
    )
    logger.info(
        f"[AGENT_SESSION_STREAM] Found {len(new_parent_messages)} new parent messages"
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
        logger.info("[AGENT_SESSION_STREAM] Saved bulk update message")

    # Build LLM context
    logger.info("[AGENT_SESSION_STREAM] ===== Building LLM Context =====")
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
    )
    logger.info("[AGENT_SESSION_STREAM] LLM context built")

    # Run with streaming
    logger.info(
        "[AGENT_SESSION_STREAM] ===== Running PromptSessionRuntime (streaming) ====="
    )
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
    update_count = 0
    logger.info("[AGENT_SESSION_STREAM] Starting runtime.run() loop...")

    async for update in runtime.run():
        update_count += 1
        yield update

        if (
            update.type == UpdateType.TOOL_COMPLETE
            and update.tool_name == "post_to_chatroom"
        ):
            posted_to_parent = True
            logger.info(
                f"[AGENT_SESSION_STREAM] >>> {actor.username} posted to chatroom <<<"
            )

    logger.info(
        f"[AGENT_SESSION_STREAM] Runtime loop completed, {update_count} updates yielded"
    )

    # Update sync point
    if new_parent_messages:
        last_msg_id = new_parent_messages[-1].id
        agent_session.update(last_parent_message_id=last_msg_id)
        logger.info(
            f"[AGENT_SESSION_STREAM] Updated sync point to message {last_msg_id}"
        )

    if posted_to_parent:
        logger.info(
            f"[AGENT_SESSION_STREAM] Turn completed: {actor.username} posted to chatroom"
        )
    else:
        logger.warning(
            f"[AGENT_SESSION_STREAM] Turn completed: {actor.username} did NOT post to chatroom"
        )

    logger.info(
        "[AGENT_SESSION_STREAM] ========== run_agent_session_turn_streaming END =========="
    )
