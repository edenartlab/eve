import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pytz
import sentry_sdk
from apscheduler.triggers.cron import CronTrigger
from bson import ObjectId

from eve.agent import Agent
from eve.agent.session.models import EdenMessageData, EdenMessageType, Session
from eve.api.api_requests import RunTriggerRequest
from eve.api.errors import APIError, handle_errors
from eve.mongo import Collection, Document
from eve.user import User

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()

# Trigger retry configuration constants
MAX_ERROR_COUNT = 2
RETRY_DELAY_MINUTES = 5
STUCK_TRIGGER_THRESHOLD_MINUTES = 90


@Collection("triggers2")
class Trigger(Document):
    """
    Trigger model for scheduled prompts.

    Triggers are tied to sessions (not agents). The agent is derived
    from session.agents[0] at execution time.
    """

    # Core fields
    name: str = "Untitled Task"
    prompt: str
    schedule: Optional[Dict[str, Any]] = (
        None  # Optional - tasks without schedules run manually only
    )
    user: ObjectId
    session: ObjectId

    # Status and execution tracking
    status: Literal["active", "paused", "running", "finished"] = "active"
    deleted: bool = False
    last_run_time: Optional[datetime] = None
    next_scheduled_run: Optional[datetime] = None

    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None

    # Run history - each run records message_id and timestamp
    runs: List[Dict[str, Any]] = []


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


def atomic_set_running(trigger_id: str) -> Optional[Trigger]:
    """Atomically set trigger status to 'running' if not already running.

    Returns:
        Trigger object if successful, None if already running
    """
    from eve.mongo import get_collection

    collection = get_collection("triggers2")
    result = collection.find_one_and_update(
        {
            "_id": ObjectId(trigger_id),
            "status": {"$ne": "running"},  # Only update if NOT running
        },
        {"$set": {"status": "running", "last_run_time": datetime.now(timezone.utc)}},
        return_document=True,
    )

    if result:
        schema = Trigger.convert_from_mongo(result)
        return Trigger.from_schema(schema, from_yaml=False)
    return None


def notify_trigger_paused(trigger: Trigger, reason: str):
    """Send in-app notification when trigger is paused."""
    try:
        from eve.mongo import get_collection

        # Create in-app notification document
        notification = {
            "user": trigger.user,
            "type": "trigger_paused",
            "title": f"Trigger '{trigger.name}' has been paused",
            "message": f"Your trigger has been automatically paused: {reason}",
            "trigger_id": trigger.id,
            "created_at": datetime.now(timezone.utc),
            "read": False,
        }

        # Store notification in notifications collection
        notifications_collection = get_collection("notifications")
        notifications_collection.insert_one(notification)

        logger.info(
            f"[NOTIFICATION] Trigger paused notification sent for trigger {trigger.id}"
        )

    except Exception as e:
        # Don't fail the whole operation if notification fails
        logger.error(
            f"[NOTIFICATION] Failed to send notification for trigger {trigger.id}: {e}"
        )
        sentry_sdk.capture_exception(e)


def _load_trigger_dependencies(trigger_id: str) -> tuple[Trigger, Session, Agent, User]:
    """Load and validate all trigger dependencies from MongoDB.

    Returns:
        Tuple of (trigger, session, agent, user)

    Raises:
        APIError: If any dependency is not found or invalid
    """
    # Load trigger
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger:
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    # Validate and load session
    if not trigger.session:
        raise APIError("Trigger has no session configured", status_code=400)

    session = Session.from_mongo(trigger.session)
    if not session:
        raise APIError(f"Session {trigger.session} not found", status_code=404)

    # Validate and load agent
    if not session.agents:
        raise APIError("Session has no agents configured", status_code=400)

    agent = Agent.from_mongo(session.agents[0])
    if not agent:
        raise APIError(f"Agent {session.agents[0]} not found", status_code=404)

    # Load user
    user = User.from_mongo(trigger.user)
    if not user:
        raise APIError(f"User {trigger.user} not found", status_code=404)

    return trigger, session, agent, user


@handle_errors
async def execute_trigger_async(
    trigger_id: str, skip_message_add: bool = False
) -> Session:
    """
    Execute a trigger asynchronously using the unified orchestrator.

    This function is designed to be spawned as a background task (e.g., via Modal).
    It assumes the trigger has already been prepared and marked as 'running'.

    It:
    1. Loads trigger, session, agent, user (trigger should already be 'running')
    2. Runs orchestrate_trigger() for the actual LLM work
    3. Handles errors with retry logic (2 failures -> pause)
    4. Calculates next scheduled run time
    """
    from datetime import timedelta

    from eve.agent.session.orchestrator import orchestrate_trigger

    logger.info(f"[TRIGGER_ASYNC] Starting async execution: trigger_id={trigger_id}")

    # Load trigger
    trigger = Trigger.from_mongo(trigger_id)
    if not trigger:
        raise APIError(f"Trigger not found: {trigger_id}", status_code=404)

    # For scheduled triggers, atomically set to running
    # For manual triggers (skip_message_add=True), status was already set by handle_trigger_run
    if not skip_message_add:
        trigger = atomic_set_running(trigger_id)
        if not trigger:
            logger.warning(
                "[TRIGGER_ASYNC] Duplicate execution prevented (already running)"
            )
            return None

    # Load dependencies
    if not trigger.session:
        raise APIError("Trigger has no session configured", status_code=400)

    session = Session.from_mongo(trigger.session)
    if not session or not session.agents:
        raise APIError(
            f"Session {trigger.session} not found or has no agents", status_code=404
        )

    agent = Agent.from_mongo(session.agents[0])
    if not agent:
        raise APIError(f"Agent {session.agents[0]} not found", status_code=404)

    user = User.from_mongo(trigger.user)
    if not user:
        raise APIError(f"User {trigger.user} not found", status_code=404)

    # Get prompt content
    prompt = trigger.prompt
    update_count = 0
    trigger_message_id = None

    try:
        # Use unified orchestrator (includes full observability)
        # If skip_message_add=True, pass None to skip message creation (message already added)
        async for update in orchestrate_trigger(
            trigger_id=str(trigger.id),
            trigger_prompt=None if skip_message_add else prompt,
            session=session,
            agent=agent,
            user_id=str(user.id),
        ):
            update_count += 1

            # Capture the trigger message ID for run tracking
            # (only relevant when not skipping message add)
            if not skip_message_add and update.get("type") == "trigger_message_created":
                trigger_message_id = update.get("message_id")

        # Record the run with message_id and timestamp
        if trigger_message_id:
            run_record = {
                "message_id": ObjectId(trigger_message_id),
                "ran_at": datetime.now(timezone.utc),
            }
            trigger.push(pushes={"runs": run_record})

        # Success - reset error count
        trigger.update(error_count=0, last_error=None)
        logger.info(f"[TRIGGER_ASYNC] Execution successful (updates: {update_count})")

    except Exception as e:
        error_msg = str(e)[:500]  # Truncate error message
        error_count = (trigger.error_count or 0) + 1

        logger.error(
            f"[TRIGGER_ASYNC] Execution failed: {type(e).__name__}: {error_msg} "
            f"(error_count: {error_count}, updates: {update_count})"
        )
        sentry_sdk.capture_exception(e)

        if error_count >= MAX_ERROR_COUNT:
            # Too many failures - pause the trigger
            logger.warning(
                f"[TRIGGER_ASYNC] Pausing trigger after {MAX_ERROR_COUNT} failures"
            )
            trigger.update(
                status="paused",
                error_count=error_count,
                last_error=error_msg,
            )
            # Send notification to user
            notify_trigger_paused(
                trigger, f"Failed 2 times in a row. Last error: {error_msg[:100]}"
            )
        else:
            # Retry after configured delay
            retry_time = datetime.now(timezone.utc) + timedelta(
                minutes=RETRY_DELAY_MINUTES
            )
            trigger.update(
                status="active",
                error_count=error_count,
                last_error=error_msg,
                next_scheduled_run=retry_time,
            )

        return session

    finally:
        # Calculate next scheduled run (only if still in running state)
        if trigger.status == "running":
            # If no schedule, task stays active (manual-only)
            if not trigger.schedule:
                trigger.update(status="active")
            else:
                next_run = calculate_next_scheduled_run(trigger.schedule)
                if next_run:
                    trigger.update(status="active", next_scheduled_run=next_run)
                    logger.info(f"[TRIGGER_ASYNC] Next run scheduled: {next_run}")
                else:
                    trigger.update(status="finished", next_scheduled_run=None)
                    logger.info(
                        "[TRIGGER_ASYNC] Trigger finished (no more scheduled runs)"
                    )

    return session


@handle_errors
async def handle_trigger_run(
    request: RunTriggerRequest,
):
    """Handle manual "Run Now" trigger execution from API.

    This function:
    1. Loads and validates trigger dependencies
    2. Atomically sets trigger to 'running' status
    3. Adds the trigger message to the session (BLOCKS until complete)
    4. Returns API response (user can now see the message)
    5. Spawns background task for agent processing
    """
    import asyncio

    from eve.agent.session.models import ChatMessage

    trigger_id = request.trigger_id
    logger.info(f"[TRIGGER_RUN] Starting manual trigger execution: {trigger_id}")

    try:
        # Load and validate all dependencies
        trigger, session, agent, user = _load_trigger_dependencies(trigger_id)
        session_id = str(session.id)

        # Atomically set status to running
        trigger = atomic_set_running(trigger_id)
        if not trigger:
            raise APIError("Trigger is already running", status_code=409)

        # Add trigger message to session INLINE (blocks API response)
        trigger_message = ChatMessage(
            role="eden",
            session=[session.id],
            sender=user.id,
            triggering_user=user.id,
            content=trigger.prompt,
            trigger=trigger.id,
            eden_message_data=EdenMessageData(message_type=EdenMessageType.TRIGGER),
        )
        trigger_message.save()
        message_id = str(trigger_message.id)

        # Record this run
        trigger.push(
            pushes={
                "runs": {
                    "message_id": ObjectId(message_id),
                    "ran_at": datetime.now(timezone.utc),
                }
            }
        )

        # NOW spawn background task for agent processing
        if os.getenv("MODAL_SERVE") == "1":
            # Production: spawn Modal function with skip_message_add=True
            try:
                import modal

                db = os.getenv("DB", "STAGE").upper()
                func = modal.Function.from_name(
                    f"api-{db.lower()}",
                    "execute_trigger_fn",
                    environment_name="main",
                )
                func.spawn(trigger_id, skip_message_add=True)
            except Exception as e:
                logger.warning(
                    f"[TRIGGER_RUN] Modal spawn failed ({e}), falling back to asyncio"
                )
                asyncio.create_task(
                    execute_trigger_async(trigger_id, skip_message_add=True)
                )
        else:
            # Local development - use asyncio.create_task
            asyncio.create_task(
                execute_trigger_async(trigger_id, skip_message_add=True)
            )

        logger.info(f"[TRIGGER_RUN] Execution started, message_id={message_id}")

        # Return immediately - message is already in session!
        return {
            "trigger_id": trigger_id,
            "session_id": session_id,
            "message_id": message_id,
            "executed": True,
        }

    except APIError:
        # Re-raise API errors (they have proper status codes)
        raise
    except Exception as e:
        logger.error(f"[TRIGGER_RUN] Failed: {type(e).__name__}: {str(e)}")
        raise APIError(f"Failed to execute trigger: {str(e)}", status_code=500)
