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
from datetime import datetime, timedelta, timezone
from typing import Optional

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


def get_first_user_message(session: Session) -> Optional[ChatMessage]:
    """Get the first user message from the parent session.

    For automatic sessions, the scenario/prompt comes from the first user message
    rather than session.context. This matches how sessions are created from the website.

    Args:
        session: The parent automatic session

    Returns:
        The first user message, or None if not found
    """
    message_doc = ChatMessage.get_collection().find_one(
        {
            "session": session.id,
            "role": "user",
        },
        sort=[("createdAt", 1)],  # Oldest first
    )
    if message_doc:
        logger.info(
            f"[AUTO] Found message_doc _id type: {type(message_doc.get('_id'))}"
        )
        logger.info(f"[AUTO] message_doc keys: {list(message_doc.keys())}")
        # Just return content since we only need that for planning
        return type("SimpleMessage", (), {"content": message_doc.get("content", "")})()
    return None


async def generate_moderator_plan(
    parent_session: Session,
    first_user_message: str,
) -> str:
    """Generate comprehensive moderator instructions from user's initial message.

    This is a one-off LLM call that converts the user's simple request into
    detailed instructions for how to moderate the session. The output becomes
    the moderator's session context.

    Args:
        parent_session: The parent automatic session
        first_user_message: The user's initial request/scenario

    Returns:
        Comprehensive moderator instructions to be used as moderator_session.context
    """
    import os
    import uuid

    from eve.agent.llm.llm import async_prompt, get_provider
    from eve.agent.session.models import (
        LLMConfig,
        LLMContext,
        LLMContextMetadata,
        LLMTraceMetadata,
    )

    logger.info("[AUTO] ===== Generating Moderator Plan =====")
    logger.info(f"[AUTO] User request: {first_user_message[:100]}...")

    # Build agent descriptions with full details
    agent_descriptions = []
    for agent_id in parent_session.agents:
        agent = Agent.from_mongo(agent_id)
        if agent:
            desc = f"- **{agent.username}**: {agent.description or 'No description'}"
            if agent.persona:
                desc += f"\n  Persona: {agent.persona[:200]}..."
            agent_descriptions.append(desc)

    agents_text = (
        "\n".join(agent_descriptions) if agent_descriptions else "No agents found."
    )

    # Build the planning prompt
    system_content = """You are planning a multi-agent session. The user has provided a request, and you need to convert it into comprehensive instructions for the session moderator.

Your output will be used as the moderator's context, so write as if you are giving instructions to the moderator (use "you" language).

Generate detailed moderator instructions that include:

1. **Scenario Understanding**: What is this session about? What's the goal?

2. **Agent Initialization**: For each participating agent, what personalized context should they receive to understand their role? Be specific about what each agent should know.

3. **Turn Order Strategy**: How should agents take turns? Is there a specific order or should it be dynamic based on the scenario?

4. **Key Milestones**: What are the key phases or milestones of this session?

5. **Finish Criteria**: When should the session end? What constitutes success?

6. **Special Considerations**: Any rules, constraints, or special handling needed?

Be concise but thorough. The moderator will use these instructions to orchestrate the session."""

    user_content = f"""USER'S REQUEST:
{first_user_message}

PARTICIPATING AGENTS:
{agents_text}

Generate the moderator instructions now."""

    # Create ChatMessage objects
    system_message = ChatMessage(
        session=[parent_session.id],
        sender=ObjectId("000000000000000000000000"),
        role="system",
        content=system_content,
    )

    user_message = ChatMessage(
        session=[parent_session.id],
        sender=ObjectId("000000000000000000000000"),
        role="user",
        content=user_content,
    )

    # Create LLM context
    llm_context = LLMContext(
        messages=[system_message, user_message],
        tools=[],
        config=LLMConfig(
            model="claude-sonnet-4-20250514",
            fallback_models=["gpt-4o"],
        ),
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(parent_session.id)}",
            trace_name="moderator_planning",
            trace_id=str(uuid.uuid4()),
            generation_name="moderator_planning",
            trace_metadata=LLMTraceMetadata(
                session_id=str(parent_session.id),
            ),
        ),
        enable_tracing=False,
    )

    # Make the one-off LLM call
    logger.info("[AUTO] Making planning LLM call...")
    provider = get_provider(llm_context)
    if provider is None:
        raise RuntimeError("No LLM provider available for moderator planning")

    result = await async_prompt(llm_context, provider)

    # Extract the text response
    plan = ""
    if hasattr(result, "content") and result.content:
        # result.content could be a string or a list of blocks
        if isinstance(result.content, str):
            plan = result.content
        else:
            for block in result.content:
                if hasattr(block, "text"):
                    plan += block.text

    if not plan:
        logger.warning("[AUTO] Planning LLM returned empty response, using fallback")
        plan = first_user_message  # Fallback to original message

    logger.info(f"[AUTO] Generated moderator plan ({len(plan)} chars)")
    logger.info(f"[AUTO] Plan preview: {plan[:200]}...")

    return plan


def is_running_on_modal() -> bool:
    """Check if we're running on Modal."""
    return os.getenv("MODAL_SERVE") == "1"


def get_reply_delay(session: Session) -> int:
    """Get the reply delay for an automatic session.

    Uses session.settings.delay_interval if set, otherwise DEFAULT_REPLY_DELAY.
    A delay of 0 means instant (no delay).
    """
    if not session.settings:
        logger.info(
            f"[AUTO] get_reply_delay: no settings, using default {DEFAULT_REPLY_DELAY}"
        )
        return DEFAULT_REPLY_DELAY

    # Handle both dict (from MongoDB) and SessionSettings object
    if isinstance(session.settings, dict):
        delay = session.settings.get("delay_interval")
    else:
        delay = getattr(session.settings, "delay_interval", None)

    logger.info(
        f"[AUTO] get_reply_delay: settings={session.settings}, delay_interval={delay}"
    )

    # Return delay if explicitly set (including 0), otherwise default
    if delay is not None:
        return delay
    return DEFAULT_REPLY_DELAY


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

    # Only sleep and set waiting_until if there's actually a delay
    if delay_seconds > 0:
        # Set waiting_until so client knows we're in a delay period
        try:
            session = Session.from_mongo(session_id)
            if session:
                wait_until = datetime.now(timezone.utc) + timedelta(
                    seconds=delay_seconds
                )
                session.update(waiting_until=wait_until)
                logger.info(
                    f"[AUTO] >>>>>> SLEEPING FOR {delay_seconds} SECONDS (until {wait_until.isoformat()}) <<<<<<"
                )
        except Exception as e:
            logger.warning(f"[AUTO] Could not set waiting_until: {e}")

        await asyncio.sleep(delay_seconds)
        logger.info(
            f"[AUTO] >>>>>> DELAY COMPLETE ({delay_seconds}s), WAKING UP <<<<<<"
        )

    logger.info(f"[AUTO] _delayed_automatic_run proceeding for {session_id}")

    try:
        session = Session.from_mongo(session_id)
        logger.info(
            f"[AUTO] Loaded session {session_id}, status={session.status}, type={session.session_type}"
        )
        # Clear waiting_until since delay is complete
        if session.waiting_until:
            session.update(waiting_until=None)
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

    # Get delay for logging
    delay = get_reply_delay(session)
    logger.info("[AUTO] ========== run_automatic_session_step START ==========")
    logger.info(
        f"[AUTO] >>>>>> NEW MODERATOR TURN STARTING (delay_interval={delay}s) <<<<<<"
    )
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

    if session.status in ("paused", "archived", "finished"):
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

            # Get the first user message (this is the user's scenario/request)
            first_message = get_first_user_message(session)
            if first_message:
                logger.info(
                    f"[AUTO] Found first user message: {first_message.content[:100]}..."
                )
                # Use the first user message content as the moderator plan
                # (Eventually we can use generate_moderator_plan to expand this)
                moderator_plan = first_message.content

            elif session.context:
                # Fallback to session.context for backwards compatibility
                logger.info("[AUTO] No first user message, using session.context")
                moderator_plan = session.context
            else:
                raise Exception(
                    "Automatic session requires either a first user message or session.context"
                )

            # Create moderator session with the generated plan
            moderator_session_id = create_moderator_session(session, moderator_plan)
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
            instruction = "Initialize the session. Call start_session with appropriate contexts for each agent, then prompt the first agent to begin."
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
        if session.status == "finished":
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

                # Force finish if moderator didn't finish
                session.reload()
                if session.status != "finished":
                    logger.warning(
                        "[AUTO] Moderator didn't finish after budget exhaustion, forcing finish"
                    )
                    session.update(status="finished")

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

        # Schedule next step immediately (delay is now applied inside prompt_agent)
        logger.info("[AUTO] ===== Next Step Scheduling =====")
        session.reload()
        logger.info(
            f"[AUTO] Current status: {session.status}, type: {session.session_type}"
        )

        if session.status == "active" and session.session_type == "automatic":
            # No delay here - delay is applied after each prompt_agent call
            logger.info("[AUTO] Scheduling next moderator turn immediately")
            await schedule_next_automatic_run(str(session.id), delay_seconds=0)
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

    if session.status == "finished":
        logger.error(f"[AUTO] Session {session_id} is finished and cannot be restarted")
        raise ValueError(
            f"Session {session_id} is finished. Create a new session instead."
        )

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
