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

from loguru import logger

from eve.agent.session.models import Session

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

    On Modal: Uses asyncio.create_task (could be upgraded to Modal triggers)
    Locally: Uses asyncio.create_task with sleep
    """
    logger.info(f"[AUTO] Scheduling next run for {session_id} in {delay_seconds}s")
    asyncio.create_task(_delayed_automatic_run(session_id, delay_seconds))


async def _delayed_automatic_run(session_id: str, delay_seconds: int) -> None:
    """Wait for delay, then run the next step if session is still active."""
    await asyncio.sleep(delay_seconds)

    try:
        session = Session.from_mongo(session_id)
    except Exception as e:
        logger.error(f"[AUTO] Failed to load session {session_id}: {e}")
        return

    # Only run if session is active (user hasn't paused/archived)
    if session.status != "active":
        logger.info(
            f"[AUTO] Session {session_id} status is '{session.status}', stopping"
        )
        return

    if session.session_type != "automatic":
        logger.info(f"[AUTO] Session {session_id} is not automatic, stopping")
        return

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
    from fastapi import BackgroundTasks

    from eve.agent.session.conductor import conductor_select_actor
    from eve.agent.session.service import PromptSessionHandle

    logger.info(f"[AUTO] Running step for session {session.id}")

    # Prevent duplicate handling
    if session.status == "running":
        logger.info(f"[AUTO] Session {session.id} already running, skipping")
        return

    if session.status in ("paused", "archived"):
        logger.info(f"[AUTO] Session {session.id} is {session.status}, not running")
        return

    # Mark as running to prevent duplicates
    session.update(status="running")

    try:
        # Select next actor via conductor
        actor = await conductor_select_actor(session)
        logger.info(f"[AUTO] Conductor selected: {actor.username}")

        # Ensure agent_sessions exist for multi-agent sessions
        if len(session.agents) > 1 and not session.agent_sessions:
            logger.info(
                f"[AUTO] Creating agent_sessions for multi-agent session {session.id}"
            )
            from eve.agent.session.setup import create_agent_sessions

            agent_sessions = create_agent_sessions(session, session.agents)
            session.update(agent_sessions=agent_sessions)
            session.agent_sessions = agent_sessions
            logger.info(f"[AUTO] Created agent_sessions: {list(agent_sessions.keys())}")

        # Check if this is a multi-agent session with agent_sessions
        if session.agent_sessions and str(actor.id) in session.agent_sessions:
            # Use agent_session flow - hand off to agent's private workspace
            from eve.agent.session.agent_session_runtime import run_agent_session_turn

            agent_session_id = session.agent_sessions[str(actor.id)]
            logger.info(
                f"[AUTO] Handing off to agent_session {agent_session_id} "
                f"for {actor.username}"
            )

            await run_agent_session_turn(
                parent_session=session,
                agent_session_id=agent_session_id,
                actor=actor,
            )

        else:
            # Fall back to existing direct flow (single agent or no agent_sessions)
            logger.info(
                f"[AUTO] Using direct flow for {actor.username} "
                f"(agents={len(session.agents)}, agent_sessions={bool(session.agent_sessions)})"
            )
            from eve.agent.session.models import (
                ChatMessageRequestInput,
                PromptSessionContext,
            )

            # Build context without a user message (conductor picks speaker)
            context = PromptSessionContext(
                session=session,
                initiating_user_id=str(session.owner),
                message=ChatMessageRequestInput(role="system", content=""),
                actor_agent_ids=[str(actor.id)],  # Use the conductor-selected actor
            )

            handle = PromptSessionHandle(
                session=session,
                context=context,
                background_tasks=BackgroundTasks(),
            )

            # Run orchestration (don't add message - automatic sessions don't have user input)
            async for update in handle.run_orchestration(stream=False):
                logger.debug(f"[AUTO] Update: {update}")

        # Success - set back to active
        session.reload()
        if session.status == "running":
            session.update(status="active")

        # Schedule next step
        session.reload()
        if session.status == "active" and session.session_type == "automatic":
            delay = get_reply_delay(session)
            await schedule_next_automatic_run(str(session.id), delay)

    except Exception as e:
        logger.error(f"[AUTO] Error in session {session.id}: {e}")
        # Pause on error to prevent infinite error loops
        session.update(status="paused")
        raise


async def start_automatic_session(session_id: str) -> None:
    """Start an automatic session when user sets status to 'active'.

    Called by API handler when session status changes to 'active'.
    """
    session = Session.from_mongo(session_id)

    if session.session_type != "automatic":
        raise ValueError(f"Session {session_id} is not an automatic session")

    if session.status != "active":
        raise ValueError(
            f"Session {session_id} status is '{session.status}', expected 'active'"
        )

    # Run the first step immediately
    await run_automatic_session_step(session)
