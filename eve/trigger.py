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
    schedule: Dict[str, Any]
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
async def execute_trigger(trigger_id: str) -> Session:
    """
    Execute a trigger using the unified orchestrator.

    This simplified implementation:
    1. Loads trigger, session, and derives agent from session.agents[0]
    2. Uses orchestrate_trigger() for full observability
    3. Implements error handling with retry logic (3 failures -> pause)
    """
    from datetime import timedelta

    from eve.agent.session.orchestrator import orchestrate_trigger

    logger.info("[TRIGGER] ========== execute_trigger START ==========")
    logger.info(f"[TRIGGER] Trigger ID: {trigger_id}")

    # Load trigger
    logger.info("[TRIGGER] Loading trigger from MongoDB...")
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger:
        logger.error(f"[TRIGGER] Trigger not found: {trigger_id}")
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    logger.info(
        f"[TRIGGER] Loaded trigger: name='{trigger.name}', status={trigger.status}"
    )
    logger.info(f"[TRIGGER] Schedule: {trigger.schedule}")
    logger.info(
        f"[TRIGGER] Error count: {trigger.error_count}, Last error: {trigger.last_error}"
    )
    logger.info(
        f"[TRIGGER] Last run: {trigger.last_run_time}, Next scheduled: {trigger.next_scheduled_run}"
    )

    # Skip if not active
    if trigger.status in ["finished", "paused"]:
        logger.info(f"[TRIGGER] Status is '{trigger.status}', skipping execution")
        logger.info("[TRIGGER] ========== execute_trigger END (skipped) ==========")
        return None

    # Skip if already running (prevent duplicate execution)
    if trigger.status == "running":
        logger.warning(
            "[TRIGGER] Status is 'running' (duplicate execution attempt), skipping"
        )
        logger.info("[TRIGGER] ========== execute_trigger END (skipped) ==========")
        return None

    # Validate session exists
    logger.info(f"[TRIGGER] Session ID from trigger: {trigger.session}")
    if not trigger.session:
        logger.error("[TRIGGER] Trigger has no session configured, skipping")
        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return None

    # Load session
    logger.info("[TRIGGER] Loading session from MongoDB...")
    session = Session.from_mongo(trigger.session)
    if not session:
        logger.error(f"[TRIGGER] Session {trigger.session} not found")
        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return None

    logger.info(f"[TRIGGER] Loaded session: id={session.id}, title='{session.title}'")
    logger.info(f"[TRIGGER] Session agents: {session.agents}")

    # Derive agent from session (not from trigger.agent)
    if not session.agents:
        logger.error("[TRIGGER] Session has no agents configured")
        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return None

    logger.info(f"[TRIGGER] Loading agent {session.agents[0]} from MongoDB...")
    agent = Agent.from_mongo(session.agents[0])
    if not agent:
        logger.error(f"[TRIGGER] Agent {session.agents[0]} not found")
        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return None

    logger.info(f"[TRIGGER] Loaded agent: id={agent.id}, username='{agent.username}'")

    # Load user
    logger.info(f"[TRIGGER] Loading user {trigger.user} from MongoDB...")
    user = User.from_mongo(trigger.user)
    if not user:
        logger.error(f"[TRIGGER] User {trigger.user} not found")
        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return None

    logger.info(f"[TRIGGER] Loaded user: id={user.id}")

    # Mark as running
    current_time = datetime.now(timezone.utc)
    logger.info(f"[TRIGGER] Marking trigger as 'running', last_run_time={current_time}")
    trigger.update(status="running", last_run_time=current_time)

    # Get prompt content
    prompt = trigger.effective_prompt
    logger.info(f"[TRIGGER] Effective prompt (first 200 chars): {prompt[:200]}...")

    logger.info("[TRIGGER] ===== Calling orchestrate_trigger =====")
    update_count = 0

    try:
        # Use unified orchestrator (includes full observability)
        async for _ in orchestrate_trigger(
            trigger_id=str(trigger.id),
            trigger_prompt=prompt,
            session=session,
            agent=agent,
            user_id=str(user.id),
        ):
            update_count += 1

        logger.info("[TRIGGER] ===== orchestrate_trigger completed =====")
        logger.info(f"[TRIGGER] Total updates received: {update_count}")

        # Success - reset error count
        logger.info("[TRIGGER] Execution successful, resetting error_count to 0")
        trigger.update(error_count=0, last_error=None)

    except Exception as e:
        error_msg = str(e)[:500]  # Truncate error message
        error_count = (trigger.error_count or 0) + 1

        logger.error("[TRIGGER] !!!!! EXECUTION FAILED !!!!!")
        logger.error(f"[TRIGGER] Error type: {type(e).__name__}")
        logger.error(f"[TRIGGER] Error message: {error_msg}")
        logger.error(f"[TRIGGER] Error count after this failure: {error_count}")
        logger.error(f"[TRIGGER] Updates received before failure: {update_count}")
        sentry_sdk.capture_exception(e)

        if error_count >= 3:
            # Too many failures - pause the trigger
            logger.warning("[TRIGGER] Error count >= 3, PAUSING trigger")
            trigger.update(
                status="paused",
                error_count=error_count,
                last_error=error_msg,
            )
            # TODO: Send notification to user
        else:
            # Retry in 5 minutes
            retry_time = datetime.now(timezone.utc) + timedelta(minutes=5)
            logger.info(f"[TRIGGER] Error count < 3, scheduling retry at {retry_time}")
            trigger.update(
                status="active",
                error_count=error_count,
                last_error=error_msg,
                next_scheduled_run=retry_time,
            )

        logger.info("[TRIGGER] ========== execute_trigger END (error) ==========")
        return session

    finally:
        # Calculate next scheduled run (only if still in running state)
        logger.info(f"[TRIGGER] Finally block: current status={trigger.status}")
        if trigger.status == "running":
            logger.info("[TRIGGER] Calculating next scheduled run...")
            next_run = calculate_next_scheduled_run(trigger.schedule)
            if next_run:
                logger.info(f"[TRIGGER] Next run calculated: {next_run}")
                trigger.update(status="active", next_scheduled_run=next_run)
            else:
                logger.info("[TRIGGER] No more scheduled runs (trigger finished)")
                trigger.update(status="finished", next_scheduled_run=None)
        else:
            logger.info(
                "[TRIGGER] Status is not 'running', skipping next run calculation"
            )

    logger.info("[TRIGGER] ========== execute_trigger END (success) ==========")
    return session


@handle_errors
async def handle_trigger_run(
    request: RunTriggerRequest,
    # background_tasks: BackgroundTasks,
):
    """Handle manual "Run Now" trigger execution from API."""
    trigger_id = request.trigger_id

    logger.info("[TRIGGER_RUN] ========== handle_trigger_run START ==========")
    logger.info(f"[TRIGGER_RUN] Trigger ID: {trigger_id}")

    trigger = Trigger.from_mongo(trigger_id)

    if not trigger or trigger.deleted:
        logger.error("[TRIGGER_RUN] Trigger not found or deleted")
        raise APIError(f"Trigger {trigger_id} not found", status_code=404)

    logger.info(
        f"[TRIGGER_RUN] Loaded trigger: name='{trigger.name}', status={trigger.status}"
    )

    if trigger.status == "running":
        logger.warning("[TRIGGER_RUN] Trigger already running, rejecting request")
        raise APIError(
            f"Trigger {trigger_id} already running, try later", status_code=400
        )

    if trigger.status != "active":
        logger.warning(
            f"[TRIGGER_RUN] Trigger status is '{trigger.status}', not 'active'"
        )
        raise APIError(
            f"Trigger {trigger_id} is not active (status: {trigger.status})",
            status_code=400,
        )

    try:
        logger.info("[TRIGGER_RUN] Calling execute_trigger...")
        from eve.trigger import execute_trigger

        session = await execute_trigger(trigger_id)
        session_id = str(session.id) if session else None

        logger.info(f"[TRIGGER_RUN] Execution complete, session_id={session_id}")
        logger.info(
            "[TRIGGER_RUN] ========== handle_trigger_run END (success) =========="
        )

        return {
            "trigger_id": trigger_id,
            "session_id": session_id,
            "executed": True,
        }

    except Exception as e:
        logger.error(f"[TRIGGER_RUN] !!!!! FAILED !!!!! {type(e).__name__}: {str(e)}")
        logger.info(
            "[TRIGGER_RUN] ========== handle_trigger_run END (error) =========="
        )
        raise APIError(f"Failed to execute trigger: {str(e)}", status_code=500)
