import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pytz
import sentry_sdk
from apscheduler.triggers.cron import CronTrigger
from bson import ObjectId

from eve.agent import Agent
from eve.agent.session.models import Session
from eve.api.api_requests import RunTriggerRequest
from eve.api.errors import APIError, handle_errors
from eve.mongo import Collection, Document
from eve.user import User

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()


@Collection("triggers2")
class Trigger(Document):
    """
    Trigger model for scheduled prompts.

    Triggers are tied to sessions (not agents). The agent is derived
    from session.agents[0] at execution time.
    """

    # Core fields
    name: str = "Untitled Task"
    prompt: Optional[str] = None  # New field - preferred over trigger_prompt
    schedule: Optional[Dict[str, Any]] = (
        None  # Optional - tasks without schedules run manually only
    )
    user: ObjectId
    session: Optional[ObjectId] = (
        None  # Required for new triggers, optional for backward compat
    )

    # Status and execution tracking
    status: Literal["active", "paused", "running", "finished"] = "active"
    deleted: bool = False
    last_run_time: Optional[datetime] = None
    next_scheduled_run: Optional[datetime] = None

    # Error handling (NEW)
    error_count: int = 0
    last_error: Optional[str] = None

    # Run history - each run records message_id and timestamp
    runs: List[Dict[str, Any]] = []

    # DEPRECATED - kept for backward compatibility with existing triggers
    trigger_prompt: Optional[str] = None  # Use 'prompt' instead
    agent: Optional[ObjectId] = None  # Derived from session.agents[0]
    context: Optional[str] = None  # No longer used
    posting_instructions: Optional[List[Dict[str, Any]]] = None  # Feature removed
    session_type: Optional[Literal["new", "another"]] = "new"  # Always reuse session
    update_config: Optional[Dict[str, Any]] = None  # Feature removed

    @property
    def effective_prompt(self) -> str:
        """Get the prompt content, handling backward compatibility."""
        return self.prompt or self.trigger_prompt or ""


def calculate_next_scheduled_run(schedule: dict) -> datetime:
    """Calculate the next scheduled run time based on the cron schedule"""
    # Extract schedule parameters
    hour = schedule.get("hour", "*")
    minute = schedule.get("minute", "*")
    day_of_month = schedule.get("day_of_month") or schedule.get("day", "*")
    month = schedule.get("month", "*")
    day_of_week = schedule.get("day_of_week", "*")
    timezone_str = schedule.get("timezone", "UTC")
    end_date = schedule.get("end_date")

    # Create CronTrigger
    trigger = CronTrigger(
        hour=hour,
        minute=minute,
        day=day_of_month,
        month=month,
        day_of_week=day_of_week,
        timezone=timezone_str,
        end_date=end_date,
    )

    # Get next fire time
    next_time = trigger.get_next_fire_time(
        None, datetime.now(pytz.timezone(timezone_str))
    )
    if next_time:
        # Convert to UTC for storage
        return next_time.astimezone(pytz.UTC).replace(tzinfo=timezone.utc)
    return None


@handle_errors
async def prepare_trigger_execution(
    trigger_id: str,
) -> tuple[Trigger, Session, Agent, User, str]:
    """
    Prepare a trigger for execution - validation, loading, and status update.

    This function:
    1. Validates the trigger can run (status checks)
    2. Loads trigger, session, agent, and user from MongoDB
    3. Marks the trigger as 'running'
    4. Returns all loaded objects and the effective prompt

    Returns:
        tuple of (trigger, session, agent, user, prompt)

    Raises:
        APIError: If validation fails or required objects not found
    """
    logger.info("[TRIGGER_PREP] ========== prepare_trigger_execution START ==========")
    logger.info(f"[TRIGGER_PREP] Trigger ID: {trigger_id}")

    # Load trigger
    logger.info("[TRIGGER_PREP] Loading trigger from MongoDB...")
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger:
        logger.error(f"[TRIGGER_PREP] Trigger not found: {trigger_id}")
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    logger.info(
        f"[TRIGGER_PREP] Loaded trigger: name='{trigger.name}', status={trigger.status}"
    )
    logger.info(f"[TRIGGER_PREP] Schedule: {trigger.schedule}")
    logger.info(
        f"[TRIGGER_PREP] Error count: {trigger.error_count}, Last error: {trigger.last_error}"
    )
    logger.info(
        f"[TRIGGER_PREP] Last run: {trigger.last_run_time}, Next scheduled: {trigger.next_scheduled_run}"
    )

    # Skip if not active
    if trigger.status in ["finished", "paused"]:
        logger.info(f"[TRIGGER_PREP] Status is '{trigger.status}', cannot execute")
        raise APIError(
            f"Trigger status is '{trigger.status}', cannot execute", status_code=400
        )

    # Skip if already running (prevent duplicate execution)
    if trigger.status == "running":
        logger.warning(
            "[TRIGGER_PREP] Status is 'running' (duplicate execution attempt)"
        )
        raise APIError("Trigger is already running", status_code=409)

    # Validate session exists
    logger.info(f"[TRIGGER_PREP] Session ID from trigger: {trigger.session}")
    if not trigger.session:
        logger.error("[TRIGGER_PREP] Trigger has no session configured")
        raise APIError("Trigger has no session configured", status_code=400)

    # Load session
    logger.info("[TRIGGER_PREP] Loading session from MongoDB...")
    session = Session.from_mongo(trigger.session)
    if not session:
        logger.error(f"[TRIGGER_PREP] Session {trigger.session} not found")
        raise APIError(f"Session {trigger.session} not found", status_code=404)

    logger.info(
        f"[TRIGGER_PREP] Loaded session: id={session.id}, title='{session.title}'"
    )
    logger.info(f"[TRIGGER_PREP] Session agents: {session.agents}")

    # Derive agent from session (not from trigger.agent)
    if not session.agents:
        logger.error("[TRIGGER_PREP] Session has no agents configured")
        raise APIError("Session has no agents configured", status_code=400)

    logger.info(f"[TRIGGER_PREP] Loading agent {session.agents[0]} from MongoDB...")
    agent = Agent.from_mongo(session.agents[0])
    if not agent:
        logger.error(f"[TRIGGER_PREP] Agent {session.agents[0]} not found")
        raise APIError(f"Agent {session.agents[0]} not found", status_code=404)

    logger.info(
        f"[TRIGGER_PREP] Loaded agent: id={agent.id}, username='{agent.username}'"
    )

    # Load user
    logger.info(f"[TRIGGER_PREP] Loading user {trigger.user} from MongoDB...")
    user = User.from_mongo(trigger.user)
    if not user:
        logger.error(f"[TRIGGER_PREP] User {trigger.user} not found")
        raise APIError(f"User {trigger.user} not found", status_code=404)

    logger.info(f"[TRIGGER_PREP] Loaded user: id={user.id}")

    # Mark as running
    current_time = datetime.now(timezone.utc)
    logger.info(
        f"[TRIGGER_PREP] Marking trigger as 'running', last_run_time={current_time}"
    )
    trigger.update(status="running", last_run_time=current_time)

    # Get prompt content
    prompt = trigger.effective_prompt
    logger.info(f"[TRIGGER_PREP] Effective prompt (first 200 chars): {prompt[:200]}...")

    logger.info("[TRIGGER_PREP] ========== prepare_trigger_execution END ==========")
    return trigger, session, agent, user, prompt


@handle_errors
async def execute_trigger_async(trigger_id: str) -> Session:
    """
    Execute a trigger asynchronously using the unified orchestrator.

    This function is designed to be spawned as a background task (e.g., via Modal).
    It assumes the trigger has already been prepared and marked as 'running'.

    It:
    1. Loads trigger, session, agent, user (trigger should already be 'running')
    2. Runs orchestrate_trigger() for the actual LLM work
    3. Handles errors with retry logic (3 failures -> pause)
    4. Calculates next scheduled run time
    """
    from datetime import timedelta

    from eve.agent.session.orchestrator import orchestrate_trigger

    logger.info("[TRIGGER_ASYNC] ========== execute_trigger_async START ==========")
    logger.info(f"[TRIGGER_ASYNC] Trigger ID: {trigger_id}")

    # Load trigger (should already be marked as 'running')
    logger.info("[TRIGGER_ASYNC] Loading trigger from MongoDB...")
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger:
        logger.error(f"[TRIGGER_ASYNC] Trigger not found: {trigger_id}")
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    logger.info(
        f"[TRIGGER_ASYNC] Loaded trigger: name='{trigger.name}', status={trigger.status}"
    )

    # Skip if not active or running
    if trigger.status not in ["active", "running"]:
        logger.info(
            f"[TRIGGER_ASYNC] Status is '{trigger.status}', cannot execute, skipping"
        )
        return None

    # Mark as running if not already (for scheduler path)
    if trigger.status == "active":
        current_time = datetime.now(timezone.utc)
        logger.info(
            f"[TRIGGER_ASYNC] Marking trigger as 'running', last_run_time={current_time}"
        )
        trigger.update(status="running", last_run_time=current_time)
    else:
        logger.info(
            "[TRIGGER_ASYNC] Status already 'running' (prepared by handle_trigger_run)"
        )

    # Validate session exists
    if not trigger.session:
        logger.error("[TRIGGER_ASYNC] Trigger has no session configured")
        raise APIError("Trigger has no session configured", status_code=400)

    # Load session
    logger.info("[TRIGGER_ASYNC] Loading session from MongoDB...")
    session = Session.from_mongo(trigger.session)
    if not session:
        logger.error(f"[TRIGGER_ASYNC] Session {trigger.session} not found")
        raise APIError(f"Session {trigger.session} not found", status_code=404)

    logger.info(
        f"[TRIGGER_ASYNC] Loaded session: id={session.id}, title='{session.title}'"
    )

    # Load agent
    if not session.agents:
        logger.error("[TRIGGER_ASYNC] Session has no agents configured")
        raise APIError("Session has no agents configured", status_code=400)

    logger.info(f"[TRIGGER_ASYNC] Loading agent {session.agents[0]} from MongoDB...")
    agent = Agent.from_mongo(session.agents[0])
    if not agent:
        logger.error(f"[TRIGGER_ASYNC] Agent {session.agents[0]} not found")
        raise APIError(f"Agent {session.agents[0]} not found", status_code=404)

    logger.info(
        f"[TRIGGER_ASYNC] Loaded agent: id={agent.id}, username='{agent.username}'"
    )

    # Load user
    logger.info(f"[TRIGGER_ASYNC] Loading user {trigger.user} from MongoDB...")
    user = User.from_mongo(trigger.user)
    if not user:
        logger.error(f"[TRIGGER_ASYNC] User {trigger.user} not found")
        raise APIError(f"User {trigger.user} not found", status_code=404)

    logger.info(f"[TRIGGER_ASYNC] Loaded user: id={user.id}")

    # Get prompt content
    prompt = trigger.effective_prompt
    logger.info(
        f"[TRIGGER_ASYNC] Effective prompt (first 200 chars): {prompt[:200]}..."
    )

    logger.info("[TRIGGER_ASYNC] ===== Calling orchestrate_trigger =====")
    update_count = 0
    trigger_message_id = None

    try:
        # Use unified orchestrator (includes full observability)
        async for update in orchestrate_trigger(
            trigger_id=str(trigger.id),
            trigger_prompt=prompt,
            session=session,
            agent=agent,
            user_id=str(user.id),
        ):
            update_count += 1

            # Capture the trigger message ID for run tracking
            if update.get("type") == "trigger_message_created":
                trigger_message_id = update.get("message_id")
                logger.info(
                    f"[TRIGGER_ASYNC] Captured trigger message_id: {trigger_message_id}"
                )

        logger.info("[TRIGGER_ASYNC] ===== orchestrate_trigger completed =====")
        logger.info(f"[TRIGGER_ASYNC] Total updates received: {update_count}")

        # Record the run with message_id and timestamp
        if trigger_message_id:
            run_record = {
                "message_id": ObjectId(trigger_message_id),
                "ran_at": datetime.now(timezone.utc),
            }
            # Append to runs array using push method
            trigger.push(pushes={"runs": run_record})
            logger.info(
                f"[TRIGGER_ASYNC] Recorded run: message_id={trigger_message_id}"
            )

        # Success - reset error count
        logger.info("[TRIGGER_ASYNC] Execution successful, resetting error_count to 0")
        trigger.update(error_count=0, last_error=None)

    except Exception as e:
        error_msg = str(e)[:500]  # Truncate error message
        error_count = (trigger.error_count or 0) + 1

        logger.error("[TRIGGER_ASYNC] !!!!! EXECUTION FAILED !!!!!")
        logger.error(f"[TRIGGER_ASYNC] Error type: {type(e).__name__}")
        logger.error(f"[TRIGGER_ASYNC] Error message: {error_msg}")
        logger.error(f"[TRIGGER_ASYNC] Error count after this failure: {error_count}")
        logger.error(f"[TRIGGER_ASYNC] Updates received before failure: {update_count}")
        sentry_sdk.capture_exception(e)

        if error_count >= 3:
            # Too many failures - pause the trigger
            logger.warning("[TRIGGER_ASYNC] Error count >= 3, PAUSING trigger")
            trigger.update(
                status="paused",
                error_count=error_count,
                last_error=error_msg,
            )
            # TODO: Send notification to user
        else:
            # Retry in 5 minutes
            retry_time = datetime.now(timezone.utc) + timedelta(minutes=5)
            logger.info(
                f"[TRIGGER_ASYNC] Error count < 3, scheduling retry at {retry_time}"
            )
            trigger.update(
                status="active",
                error_count=error_count,
                last_error=error_msg,
                next_scheduled_run=retry_time,
            )

        logger.info(
            "[TRIGGER_ASYNC] ========== execute_trigger_async END (error) =========="
        )
        return session

    finally:
        # Calculate next scheduled run (only if still in running state)
        logger.info(f"[TRIGGER_ASYNC] Finally block: current status={trigger.status}")
        if trigger.status == "running":
            # If no schedule, task stays active (manual-only)
            if not trigger.schedule:
                logger.info(
                    "[TRIGGER_ASYNC] No schedule - task stays active (manual-only)"
                )
                trigger.update(status="active")
            else:
                logger.info("[TRIGGER_ASYNC] Calculating next scheduled run...")
                next_run = calculate_next_scheduled_run(trigger.schedule)
                if next_run:
                    logger.info(f"[TRIGGER_ASYNC] Next run calculated: {next_run}")
                    trigger.update(status="active", next_scheduled_run=next_run)
                else:
                    logger.info(
                        "[TRIGGER_ASYNC] No more scheduled runs (trigger finished)"
                    )
                    trigger.update(status="finished", next_scheduled_run=None)
        else:
            logger.info(
                "[TRIGGER_ASYNC] Status is not 'running', skipping next run calculation"
            )

    logger.info(
        "[TRIGGER_ASYNC] ========== execute_trigger_async END (success) =========="
    )
    return session


@handle_errors
async def handle_trigger_run(
    request: RunTriggerRequest,
):
    """Handle manual "Run Now" trigger execution from API.

    This function returns immediately after preparing the trigger and spawning
    the background execution task. The actual agent prompt runs asynchronously.
    """
    import asyncio

    trigger_id = request.trigger_id

    logger.info("[TRIGGER_RUN] ========== handle_trigger_run START ==========")
    logger.info(f"[TRIGGER_RUN] Trigger ID: {trigger_id}")

    try:
        # Prepare execution (validates, loads, marks as running)
        # This is fast and returns the session immediately
        logger.info("[TRIGGER_RUN] Preparing trigger execution...")
        trigger, session, agent, user, prompt = await prepare_trigger_execution(
            trigger_id
        )
        session_id = str(session.id)

        logger.info(f"[TRIGGER_RUN] Preparation complete, session_id={session_id}")
        logger.info("[TRIGGER_RUN] Spawning async execution task...")

        # Spawn execution based on environment
        if os.getenv("MODAL_SERVE") == "1":
            # Production: spawn Modal function
            try:
                import modal

                logger.info(
                    "[TRIGGER_RUN] Running on Modal, using execute_trigger_fn.spawn()"
                )
                db = os.getenv("DB", "STAGE").upper()
                func = modal.Function.from_name(
                    f"api-{db.lower()}",
                    "execute_trigger_fn",
                    environment_name="main",
                )
                func.spawn(trigger_id)
            except Exception as e:
                logger.warning(
                    f"[TRIGGER_RUN] Modal spawn failed ({e}), falling back to asyncio.create_task()"
                )
                asyncio.create_task(execute_trigger_async(trigger_id))
        else:
            # Local development - use asyncio.create_task
            logger.info("[TRIGGER_RUN] Running locally, using asyncio.create_task()")
            asyncio.create_task(execute_trigger_async(trigger_id))

        logger.info("[TRIGGER_RUN] Background task spawned successfully")
        logger.info(
            "[TRIGGER_RUN] ========== handle_trigger_run END (success) =========="
        )

        return {
            "trigger_id": trigger_id,
            "session_id": session_id,
            "executed": True,
        }

    except APIError:
        # Re-raise API errors (they have proper status codes)
        logger.error("[TRIGGER_RUN] APIError during preparation")
        logger.info(
            "[TRIGGER_RUN] ========== handle_trigger_run END (error) =========="
        )
        raise
    except Exception as e:
        logger.error(f"[TRIGGER_RUN] !!!!! FAILED !!!!! {type(e).__name__}: {str(e)}")
        logger.info(
            "[TRIGGER_RUN] ========== handle_trigger_run END (error) =========="
        )
        raise APIError(f"Failed to execute trigger: {str(e)}", status_code=500)
