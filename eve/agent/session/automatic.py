"""Automatic session management.

Handles scheduling and execution of automatic sessions that run continuously
with a configurable delay between each orchestration step.

Status flow:
  - "paused" (initial): User hasn't started the session yet
  - "active": User has activated, waiting for next scheduled step
  - "running": Currently executing an orchestration step (prevents duplicates)
  - "archived": Session is done

The loop:
  1. User sets status to "active" via API
  2. run_automatic_session_step() sets status to "running"
  3. Conductor selects next speaker, agent generates response
  4. Status set back to "active", next step scheduled after delay
  5. Repeat until user pauses or archives
"""

import asyncio
import os
import traceback

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.models import (
    ChatMessage,
    EdenMessageData,
    EdenMessageType,
    Session,
)

# Default delay between automatic session steps (seconds)
DEFAULT_REPLY_DELAY = 60


def is_running_on_modal() -> bool:
    """Check if we're running on Modal."""
    return os.getenv("MODAL_SERVE") == "1"


def get_reply_delay(session: Session) -> int:
    """Get the reply delay for an automatic session.

    Uses session.settings.delay_interval if set, otherwise DEFAULT_REPLY_DELAY.
    """
    if not session.settings:
        return DEFAULT_REPLY_DELAY

    # Handle both dict (from MongoDB) and SessionSettings object
    if isinstance(session.settings, dict):
        delay = session.settings.get("delay_interval", 0)
    else:
        delay = session.settings.delay_interval

    return delay if delay and delay > 0 else DEFAULT_REPLY_DELAY


async def schedule_next_automatic_run(session_id: str, delay_seconds: int) -> None:
    """Schedule the next automatic session run after a delay.

    On Modal: Sleep in-process then continue (Modal function stays alive)
    Locally: Uses asyncio.create_task with sleep
    """
    logger.info(f"[AUTO] Scheduling next run for {session_id} in {delay_seconds}s")

    if is_running_on_modal():
        # On Modal: sleep inline and then run the next step
        # This keeps the Modal function alive for the entire session
        logger.info(f"[AUTO] Modal mode: sleeping {delay_seconds}s inline")
        await _delayed_automatic_run(session_id, delay_seconds)
    else:
        # Locally: use create_task for non-blocking behavior
        asyncio.create_task(_delayed_automatic_run(session_id, delay_seconds))


async def _delayed_automatic_run(session_id: str, delay_seconds: int) -> None:
    """Wait for delay, then run the next step if session is still active."""
    logger.info(
        f"[AUTO] _delayed_automatic_run starting for {session_id}, delay={delay_seconds}s"
    )
    await asyncio.sleep(delay_seconds)
    logger.info(f"[AUTO] _delayed_automatic_run woke up for {session_id}")

    try:
        session = Session.from_mongo(session_id)
        logger.info(
            f"[AUTO] Loaded session {session_id}, status={session.status}, type={session.session_type}"
        )
    except Exception as e:
        logger.error(
            f"[AUTO] Failed to load session {session_id}: {e}\n{traceback.format_exc()}"
        )
        return

    # Only run if session is active (user hasn't paused/archived)
    if session.status != "active":
        logger.info(
            f"[AUTO] Session {session_id} status is '{session.status}', expected 'active' - stopping loop"
        )
        return

    if session.session_type != "automatic":
        logger.info(
            f"[AUTO] Session {session_id} is not automatic (type={session.session_type}), stopping"
        )
        return

    logger.info(f"[AUTO] Proceeding to run_automatic_session_step for {session_id}")
    await run_automatic_session_step(session)


async def run_automatic_session_step(session: Session) -> None:
    """Execute one step of an automatic session.

    Flow:
    1. Set status to "running" (prevents duplicate handling)
    2. Use conductor to select next actor
    3. If multi-agent with agent_sessions: hand off to agent's private workspace
    4. Otherwise: run prompt session directly for that actor
    5. Set status back to "active"
    6. Schedule next step after delay

    For multi-agent sessions with agent_sessions:
    - Each agent has their own private workspace (agent_session)
    - The selected agent receives bulk message updates from parent
    - Agent does private work then posts to parent via post_to_chatroom tool
    """
    from eve.agent.session.conductor import conductor_select_actor

    logger.info("[AUTO] ========== run_automatic_session_step START ==========")
    logger.info(f"[AUTO] Session ID: {session.id}")
    logger.info(f"[AUTO] Session title: {session.title}")
    logger.info(f"[AUTO] Session status: {session.status}")
    logger.info(f"[AUTO] Session type: {session.session_type}")
    logger.info(f"[AUTO] Session agents: {session.agents}")
    logger.info(f"[AUTO] Session agent_sessions: {session.agent_sessions}")

    # Prevent duplicate handling
    if session.status == "running":
        logger.warning(
            "[AUTO] Session already 'running' - possible race condition, skipping"
        )
        logger.info(
            "[AUTO] ========== run_automatic_session_step END (skipped) =========="
        )
        return

    if session.status in ("paused", "archived"):
        logger.info(f"[AUTO] Session status is '{session.status}', not running step")
        logger.info(
            "[AUTO] ========== run_automatic_session_step END (skipped) =========="
        )
        return

    # Mark as running to prevent duplicates
    logger.info("[AUTO] Marking session status as 'running'")
    session.update(status="running")

    try:
        # Select next actor via conductor
        logger.info("[AUTO] ===== Conductor Selection =====")
        logger.info("[AUTO] Calling conductor_select_actor...")
        actor, conductor_response = await conductor_select_actor(session)
        logger.info(
            f"[AUTO] Conductor selected actor: {actor.username} (id={actor.id})"
        )
        logger.info(f"[AUTO] Conductor hint: {conductor_response.hint}")
        logger.info(f"[AUTO] Conductor finish: {conductor_response.finish}")

        # Check if conductor wants to finish the session
        if conductor_response.finish:
            logger.info("[AUTO] Conductor requested session finish")
            from eve.agent.session.conductor import conductor_finish_session
            from eve.agent.session.setup import create_eden_message_json

            # Generate summary and save as CONDUCTOR_FINISH eden message
            finish_response = await conductor_finish_session(session)
            create_eden_message_json(
                session_id=session.id,
                message_type=EdenMessageType.CONDUCTOR_FINISH,
                content=finish_response.model_dump_json(),
            )
            # Pause the session
            session.update(status="paused")
            logger.info("[AUTO] Session finished and paused")
            logger.info(
                "[AUTO] ========== run_automatic_session_step END (finished) =========="
            )
            return

        # Create turn notification for selected agent's workspace
        # Always send this so agent knows it's their turn and must post
        if session.agent_sessions:
            agent_session_id = session.agent_sessions.get(str(actor.id))
            if agent_session_id:
                logger.info(f"[AUTO] Creating turn notification for {actor.username}")
                # Include reminder to use post_to_chatroom
                if conductor_response.hint:
                    hint_content = f"""🎯 IT'S YOUR TURN!

{conductor_response.hint}

⚠️ IMPORTANT: You MUST use the post_to_chatroom tool to respond. Your response will not be seen by others unless you post it."""
                else:
                    hint_content = """🎯 IT'S YOUR TURN!

⚠️ IMPORTANT: You MUST use the post_to_chatroom tool to respond. Your response will not be seen by others unless you post it."""
                hint_message = ChatMessage(
                    session=[agent_session_id],  # Only to this agent's workspace
                    sender=ObjectId("000000000000000000000000"),
                    role="eden",
                    content=hint_content,
                    eden_message_data=EdenMessageData(
                        message_type=EdenMessageType.CONDUCTOR_HINT
                    ),
                )
                hint_message.save()
                logger.info("[AUTO] Turn notification message saved")

        # Ensure agent_sessions exist for multi-agent sessions
        # For automatic sessions, this should already be done via conductor init
        if len(session.agents) > 1 and not session.agent_sessions:
            logger.warning(
                f"[AUTO] Multi-agent session {session.id} missing agent_sessions. "
                "This shouldn't happen - conductor init should have created them."
            )
            # Fallback: call conductor init now
            await _initialize_conductor_for_session(session)
            session = Session.from_mongo(session.id)

        # Check if this is a multi-agent session with agent_sessions
        logger.info("[AUTO] ===== Execution Path Decision =====")
        logger.info(f"[AUTO] Has agent_sessions: {bool(session.agent_sessions)}")
        logger.info(
            f"[AUTO] Actor ID in agent_sessions: {str(actor.id) in (session.agent_sessions or {})}"
        )

        if session.agent_sessions and str(actor.id) in session.agent_sessions:
            # Use agent_session flow - hand off to agent's private workspace
            from eve.agent.session.agent_session_runtime import run_agent_session_turn

            agent_session_id = session.agent_sessions[str(actor.id)]
            logger.info("[AUTO] PATH: Multi-agent with agent_session")
            logger.info(f"[AUTO] Agent session ID: {agent_session_id}")
            logger.info("[AUTO] Calling run_agent_session_turn...")

            await run_agent_session_turn(
                parent_session=session,
                agent_session_id=agent_session_id,
                actor=actor,
            )
            logger.info(f"[AUTO] run_agent_session_turn completed for {actor.username}")

        else:
            # Fall back to existing direct flow (single agent or no agent_sessions)
            logger.info("[AUTO] PATH: Direct flow (single agent or no agent_sessions)")
            logger.info(f"[AUTO] Agent count: {len(session.agents)}")
            from eve.agent.session.orchestrator import orchestrate_automatic

            logger.info("[AUTO] Calling orchestrate_automatic...")
            update_count = 0
            async for update in orchestrate_automatic(session, actor):
                update_count += 1
                update_type = update.get("type", "unknown")
                logger.debug(f"[AUTO] Update #{update_count}: type={update_type}")

            logger.info(
                f"[AUTO] orchestrate_automatic completed, {update_count} updates"
            )

        # Increment turn counter after successful execution
        logger.info("[AUTO] ===== Turn Counter Update =====")
        if session.budget:
            new_turns = (session.budget.turns_spent or 0) + 1
            # Update the budget object and save via MongoDB set operation
            session.budget.turns_spent = new_turns
            Session.get_collection().update_one(
                {"_id": session.id},
                {"$set": {"budget.turns_spent": new_turns}},
            )
            logger.info(f"[AUTO] Turns spent: {new_turns}")

            # Check hard turn limit - finish if budget exhausted
            if session.budget.turn_budget and new_turns >= session.budget.turn_budget:
                logger.info(
                    f"[AUTO] Turn budget exhausted ({new_turns}/{session.budget.turn_budget})"
                )
                from eve.agent.session.conductor import conductor_finish_session
                from eve.agent.session.setup import create_eden_message_json

                # Generate summary and save as CONDUCTOR_FINISH eden message
                finish_response = await conductor_finish_session(session)
                create_eden_message_json(
                    session_id=session.id,
                    message_type=EdenMessageType.CONDUCTOR_FINISH,
                    content=finish_response.model_dump_json(),
                )
                # Pause the session
                session.update(status="paused")
                logger.info("[AUTO] Session finished due to turn budget exhaustion")
                logger.info(
                    "[AUTO] ========== run_automatic_session_step END (budget) =========="
                )
                return

        # Success - set back to active
        logger.info("[AUTO] ===== Post-Execution Status Update =====")
        session.reload()
        logger.info(f"[AUTO] Session status after reload: {session.status}")

        if session.status == "running":
            logger.info("[AUTO] Setting status back to 'active'")
            session.update(status="active")
        else:
            logger.warning(
                f"[AUTO] Status changed during execution to '{session.status}', not resetting to 'active'"
            )

        # Schedule next step
        logger.info("[AUTO] ===== Next Step Scheduling =====")
        session.reload()
        logger.info(
            f"[AUTO] Current status: {session.status}, type: {session.session_type}"
        )

        if session.status == "active" and session.session_type == "automatic":
            delay = get_reply_delay(session)
            logger.info(f"[AUTO] Scheduling next step in {delay}s")
            await schedule_next_automatic_run(str(session.id), delay)
        else:
            logger.info(
                f"[AUTO] NOT scheduling next step (status={session.status}, type={session.session_type})"
            )

        logger.info(
            "[AUTO] ========== run_automatic_session_step END (success) =========="
        )

    except Exception as e:
        logger.error("[AUTO] !!!!! ERROR in automatic session step !!!!!")
        logger.error(f"[AUTO] Session: {session.id}")
        logger.error(f"[AUTO] Error type: {type(e).__name__}")
        logger.error(f"[AUTO] Error message: {str(e)}")
        logger.error(f"[AUTO] Full traceback:\n{traceback.format_exc()}")
        # Pause on error to prevent infinite error loops
        logger.info("[AUTO] Pausing session to prevent infinite error loop")
        session.update(status="paused")
        logger.info(
            "[AUTO] ========== run_automatic_session_step END (error) =========="
        )
        raise


async def _initialize_conductor_for_session(session: Session) -> None:
    """Initialize conductor for a multi-agent automatic session.

    This generates personalized contexts for each agent and creates
    their private workspace sessions (agent_sessions).

    Called once when an automatic session first becomes active.
    """
    from eve.agent.session.conductor import conductor_initialize_session
    from eve.agent.session.setup import (
        create_agent_sessions_with_contexts,
        create_eden_message_json,
    )

    logger.info(f"[AUTO] ===== Conductor Initialization for {session.id} =====")

    # Load agents
    agents = {
        agent.username: agent
        for agent in [Agent.from_mongo(a) for a in session.agents]
        if agent
    }
    logger.info(f"[AUTO] Loaded agents: {list(agents.keys())}")

    # Generate personalized contexts via conductor
    logger.info("[AUTO] Calling conductor_initialize_session...")
    init_response = await conductor_initialize_session(session, agents)
    logger.info(
        f"[AUTO] Conductor generated {len(init_response.agent_contexts)} contexts"
    )
    logger.info(f"[AUTO] Finish criteria: {init_response.finish_criteria}")

    # Save as CONDUCTOR_INIT eden message
    create_eden_message_json(
        session_id=session.id,
        message_type=EdenMessageType.CONDUCTOR_INIT,
        content=init_response.model_dump_json(),
    )
    logger.info("[AUTO] Saved CONDUCTOR_INIT eden message")

    # Create agent_sessions with personalized contexts
    agent_sessions = create_agent_sessions_with_contexts(
        parent_session=session,
        agents=session.agents,
        init_response=init_response,
    )

    # Update session with agent_sessions
    session.update(agent_sessions=agent_sessions)
    logger.info(f"[AUTO] Created agent_sessions: {list(agent_sessions.keys())}")
    logger.info("[AUTO] ===== Conductor Initialization Complete =====")


async def start_automatic_session(session_id: str) -> None:
    """Start an automatic session when user sets status to 'active'.

    Called by API handler when session status changes to 'active'.
    """
    logger.info(
        f"[AUTO] ========== start_automatic_session CALLED ========== session_id={session_id}"
    )

    try:
        session = Session.from_mongo(session_id)
        logger.info(
            f"[AUTO] Loaded session: id={session.id}, status={session.status}, "
            f"type={session.session_type}, agents={len(session.agents)}"
        )
    except Exception as e:
        logger.error(
            f"[AUTO] Failed to load session {session_id}: {e}\n{traceback.format_exc()}"
        )
        raise

    if session.session_type != "automatic":
        logger.error(
            f"[AUTO] Session {session_id} is not automatic (type={session.session_type})"
        )
        raise ValueError(f"Session {session_id} is not an automatic session")

    if session.status != "active":
        logger.error(
            f"[AUTO] Session {session_id} status is '{session.status}', expected 'active'"
        )
        raise ValueError(
            f"Session {session_id} status is '{session.status}', expected 'active'"
        )

    # Initialize conductor for multi-agent sessions (if not already done)
    if len(session.agents) > 1 and not session.agent_sessions:
        logger.info(
            f"[AUTO] Multi-agent session {session_id} needs conductor initialization"
        )
        await _initialize_conductor_for_session(session)
        # Reload session to get updated agent_sessions
        session = Session.from_mongo(session_id)

    # Run the first step immediately
    logger.info(f"[AUTO] Starting first step for session {session_id}")
    await run_automatic_session_step(session)
    logger.info(
        f"[AUTO] ========== start_automatic_session COMPLETE ========== session_id={session_id}"
    )
