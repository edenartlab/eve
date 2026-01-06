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
  3. Moderator agent runs one turn (may prompt agents, conduct votes, etc.)
  4. Status set back to "active", next step scheduled after delay
  5. Repeat until moderator calls finish_session or user pauses/archives

The Moderator Agent:
  - Has its own private workspace (moderator_session)
  - Uses specialized tools: start_session, finish_session, prompt_agent, conduct_vote, chat
  - Orchestrates turn-taking, voting, and session lifecycle
  - Replaces the old conductor approach with a proper agent-based system
"""

import asyncio
import os
import traceback

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.models import (
    ChatMessage,
    Session,
)
from eve.agent.session.setup import create_moderator_session, get_moderator_agent_id

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
    """Execute one step of an automatic session using the Moderator agent.

    Flow:
    1. Set status to "running" (prevents duplicate handling)
    2. Get or create moderator_session
    3. Send "Run the next round" message to trigger the moderator
    4. Run the moderator via run_agent_session_turn
    5. Check if session was finished (status changed to "paused")
    6. Handle budget/turn counting
    7. Set status back to "active" and schedule next step

    The moderator agent:
    - On first turn: calls start_session to initialize agent workspaces
    - On subsequent turns: uses prompt_agent to select and prompt agents
    - May use conduct_vote for group decisions
    - Calls finish_session when goals are achieved
    """
    from eve.agent.session.agent_session_runtime import run_agent_session_turn

    logger.info("[AUTO] ========== run_automatic_session_step START ==========")
    logger.info(f"[AUTO] Session ID: {session.id}")
    logger.info(f"[AUTO] Session title: {session.title}")
    logger.info(f"[AUTO] Session status: {session.status}")
    logger.info(f"[AUTO] Session type: {session.session_type}")
    logger.info(f"[AUTO] Session agents: {session.agents}")
    logger.info(f"[AUTO] Session agent_sessions: {session.agent_sessions}")
    logger.info(f"[AUTO] Session moderator_session: {session.moderator_session}")

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
        # Get or create moderator_session
        if not session.moderator_session:
            logger.info("[AUTO] Creating moderator_session...")
            moderator_session_id = create_moderator_session(session)
            session.update(moderator_session=moderator_session_id)
            session.moderator_session = moderator_session_id
            logger.info(f"[AUTO] Created moderator_session: {moderator_session_id}")
        else:
            logger.info(
                f"[AUTO] Using existing moderator_session: {session.moderator_session}"
            )

        # Load moderator agent
        moderator_agent_id = get_moderator_agent_id()
        moderator_agent = Agent.from_mongo(moderator_agent_id)
        if not moderator_agent:
            raise Exception(
                f"Moderator agent not found: {moderator_agent_id}. "
                f"Ensure the moderator agent exists in the DB={os.getenv('DB', 'STAGE')} database."
            )
        logger.info(f"[AUTO] Loaded moderator agent: {moderator_agent.username}")

        # Create "Run the next round" user message to trigger the moderator
        moderator_session = Session.from_mongo(session.moderator_session)
        if not moderator_session:
            raise Exception(f"Moderator session not found: {session.moderator_session}")

        # Build instruction based on whether agent_sessions exist
        if not session.agent_sessions:
            instruction = (
                "Initialize the session. Call start_session with appropriate contexts "
                "for each agent, then prompt the first agent to begin."
            )
        else:
            instruction = (
                "Run the next round. Prompt an agent or take appropriate action."
            )

        round_message = ChatMessage(
            session=[session.moderator_session],
            sender=ObjectId(str(session.owner)),
            role="user",
            content=instruction,
        )
        round_message.save()
        logger.info(f"[AUTO] Sent moderator instruction: '{instruction}'")

        # Run the moderator's turn
        logger.info("[AUTO] Running moderator turn...")
        await run_agent_session_turn(
            parent_session=session,
            agent_session_id=session.moderator_session,
            actor=moderator_agent,
        )
        logger.info("[AUTO] Moderator turn completed")

        # Reload session to check if moderator called finish_session
        session.reload()
        if session.status == "paused":
            logger.info("[AUTO] Moderator finished the session")
            logger.info(
                "[AUTO] ========== run_automatic_session_step END (finished) =========="
            )
            return

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

            # Check hard turn limit - send budget warning to moderator
            if session.budget.turn_budget and new_turns >= session.budget.turn_budget:
                logger.info(
                    f"[AUTO] Turn budget exhausted ({new_turns}/{session.budget.turn_budget})"
                )
                # Send budget exhaustion message to moderator to trigger finish
                budget_message = ChatMessage(
                    session=[session.moderator_session],
                    sender=ObjectId("000000000000000000000000"),
                    role="user",
                    content=(
                        f"BUDGET EXHAUSTED: Turn {new_turns} of {session.budget.turn_budget}. "
                        "You must call finish_session NOW to end this session with a summary."
                    ),
                )
                budget_message.save()

                # Run one more moderator turn to finish
                await run_agent_session_turn(
                    parent_session=session,
                    agent_session_id=session.moderator_session,
                    actor=moderator_agent,
                )

                # Force pause if moderator didn't finish
                session.reload()
                if session.status != "paused":
                    logger.warning(
                        "[AUTO] Moderator didn't finish after budget exhaustion, forcing pause"
                    )
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


# NOTE: _initialize_conductor_for_session has been removed.
# The moderator agent now handles initialization via the start_session tool.
# This is called automatically on the first moderator turn when agent_sessions don't exist.


async def start_automatic_session(session_id: str) -> None:
    """Start an automatic session when user sets status to 'active'.

    Called by API handler when session status changes to 'active'.

    The moderator agent will be created and initialized on the first step.
    The moderator handles agent_session creation via the start_session tool.
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

    # Run the first step immediately
    # The moderator_session will be created on first step
    # The moderator will call start_session to initialize agent workspaces
    logger.info(f"[AUTO] Starting first step for session {session_id}")
    await run_automatic_session_step(session)
    logger.info(
        f"[AUTO] ========== start_automatic_session COMPLETE ========== session_id={session_id}"
    )
