"""Automatic session management.

Handles scheduling and execution of automatic sessions that run continuously
with a delay between each orchestration step.
"""

import asyncio
import os

from loguru import logger

from eve.agent.session.models import Session


def is_running_on_modal() -> bool:
    """Check if we're running on Modal."""
    return os.getenv("MODAL_SERVE") == "1"


async def schedule_next_automatic_run(session_id: str, delay_seconds: int = 60) -> None:
    """Schedule the next automatic session run.

    On Modal: Creates a one-shot trigger
    Locally: Uses in-process asyncio.sleep delay
    """
    if is_running_on_modal():
        await _schedule_via_trigger(session_id, delay_seconds)
    else:
        asyncio.create_task(_delayed_automatic_run(session_id, delay_seconds))


async def _schedule_via_trigger(session_id: str, delay_seconds: int) -> None:
    """Schedule next run via one-shot trigger (Modal production)."""
    # TODO: Implement one-shot trigger creation
    # For now, fall back to in-process delay
    logger.warning(
        f"One-shot trigger not yet implemented, using in-process delay for session {session_id}"
    )
    asyncio.create_task(_delayed_automatic_run(session_id, delay_seconds))


async def _delayed_automatic_run(session_id: str, delay_seconds: int) -> None:
    """In-process delayed run for local development."""
    logger.info(
        f"Scheduling next automatic run for session {session_id} in {delay_seconds}s"
    )
    await asyncio.sleep(delay_seconds)

    # Reload session to check current state
    try:
        session = Session.from_mongo(session_id)
    except Exception as e:
        logger.error(f"Failed to load session {session_id} for automatic run: {e}")
        return

    if session.status != "active":
        logger.info(
            f"Session {session_id} status is '{session.status}', skipping automatic run"
        )
        return

    if session.session_type != "automatic":
        logger.info(
            f"Session {session_id} type is '{session.session_type}', skipping automatic run"
        )
        return

    # Run the orchestration
    await run_automatic_session_step(session)


async def run_automatic_session_step(session: Session) -> None:
    """Execute one step of an automatic session.

    This runs the orchestration for the session, which will:
    1. Use the conductor to select the next actor
    2. Run the prompt session for that actor
    3. Schedule the next automatic run if session is still active
    """
    from fastapi import BackgroundTasks

    from eve.agent.session.service import PromptSessionHandle

    logger.info(f"[AUTO] Running automatic session step for {session.id}")
    logger.info(
        f"[AUTO] Session details: type={session.session_type}, status={session.status}, agents={session.agents}, owner={session.owner}"
    )

    # Check status before running
    if session.status in ("paused", "archived"):
        logger.info(f"[AUTO] Session {session.id} is {session.status}, not running")
        return

    if session.status == "running":
        logger.info(f"[AUTO] Session {session.id} is already running, skipping")
        return

    # Set status to running
    logger.info("[AUTO] Setting session status to 'running'")
    session.update(status="running")

    try:
        # Build minimal context for automatic session (no user message)
        from eve.agent.session.models import (
            ChatMessageRequestInput,
            PromptSessionContext,
        )

        logger.info(f"[AUTO] Building PromptSessionContext with owner={session.owner}")

        # For automatic sessions, we don't add a user message
        # The conductor picks the next speaker and they generate content
        context = PromptSessionContext(
            session=session,
            initiating_user_id=str(session.owner),  # Use session owner
            message=ChatMessageRequestInput(
                role="system", content=""
            ),  # Empty placeholder
        )

        logger.info("[AUTO] Creating PromptSessionHandle")

        # Create handle and run orchestration (skip add_message for automatic)
        handle = PromptSessionHandle(
            session=session,
            context=context,
            background_tasks=BackgroundTasks(),
        )

        logger.info("[AUTO] Calling handle.run_orchestration(stream=False)")

        # Run orchestration directly (don't add message for automatic sessions)
        update_count = 0
        async for update in handle.run_orchestration(stream=False):
            update_count += 1
            logger.info(
                f"[AUTO] Received update #{update_count}: {type(update).__name__} - {update}"
            )

        logger.info(f"[AUTO] Orchestration complete, received {update_count} updates")

        # Success - set status back to active
        session.reload()
        if session.status == "running":
            session.update(status="active")

        # Schedule next run if still active
        session.reload()
        if session.status == "active" and session.session_type == "automatic":
            await schedule_next_automatic_run(str(session.id), delay_seconds=60)

    except Exception as e:
        logger.error(f"Error in automatic session {session.id}: {e}")
        # On error, pause the session to prevent infinite error loops
        session.update(status="paused")
        raise


async def start_automatic_session(session_id: str) -> None:
    """Start or resume an automatic session.

    Called when a session's status changes to 'active' and session_type is 'automatic'.
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
