import copy
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
STUCK_TRIGGER_THRESHOLD_MINUTES = (
    65  # Reduced from 90 to detect timeouts faster (Modal timeout is 60min)
)


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
    session: Optional[ObjectId] = None
    agent: Optional[ObjectId] = None  # Used when creating new session if session=None

    # Subscription support
    parent_trigger: Optional[ObjectId] = (
        None  # Reference to parent if this is a subscription copy
    )

    # Status and execution tracking
    status: Literal["active", "paused", "running", "finished"] = "active"
    deleted: bool = False
    subscribable: bool = False
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
    """Atomically set trigger status to 'running' if not already running or deleted.

    Returns:
        Trigger object if successful, None if already running or deleted
    """
    from eve.mongo import get_collection

    collection = get_collection("triggers2")
    result = collection.find_one_and_update(
        {
            "_id": ObjectId(trigger_id),
            "status": {"$ne": "running"},  # Only update if NOT running
            "deleted": {"$ne": True},  # Only update if NOT deleted
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


def check_agent_access(agent_id: ObjectId, user_id: ObjectId) -> bool:
    """Check if user has access to an agent (owner, editor, or member).

    Returns:
        True if user has access, False otherwise
    """

    # For now, we ar e just making access public.
    # The code below is still valid and left for the future.
    # When asked, enforce the code below.
    return True

    # If checking permissions
    from eve.agent.agent import AgentPermission

    agent = Agent.from_mongo(agent_id)
    if not agent:
        return False

    # Check if user is agent owner
    if agent.owner == user_id:
        return True

    # Check if user has permission (owner, editor, or member)
    try:
        permission = AgentPermission.load(agent=agent_id, user=user_id)
        return permission and permission.level in ["owner", "editor", "member"]
    except Exception:
        return False


async def subscribe_to_trigger(trigger_id: str, subscriber_user_id: str) -> "Trigger":
    """Subscribe a user to a trigger, creating a child trigger copy.

    Args:
        trigger_id: ID of the parent trigger to subscribe to
        subscriber_user_id: ID of the user subscribing

    Returns:
        The newly created child trigger

    Raises:
        APIError: If validation fails
    """
    subscriber_id = ObjectId(subscriber_user_id)

    # Load and validate parent trigger
    parent = Trigger.from_mongo(trigger_id)
    if not parent:
        raise APIError(f"Trigger {trigger_id} not found", status_code=404)

    if parent.deleted:
        raise APIError("Cannot subscribe to a deleted trigger", status_code=400)

    if not parent.subscribable:
        raise APIError("This trigger does not allow subscriptions", status_code=403)

    if parent.parent_trigger is not None:
        raise APIError(
            "Cannot subscribe to a child trigger. Subscribe to the parent instead.",
            status_code=400,
        )

    # Check subscriber has agent access
    if not parent.agent:
        raise APIError("Parent trigger has no agent configured", status_code=400)

    if not check_agent_access(parent.agent, subscriber_id):
        raise APIError("You do not have permission to use this agent", status_code=403)

    # Check not already subscribed
    existing = Trigger.find_one(
        {
            "parent_trigger": parent.id,
            "user": subscriber_id,
            "deleted": {"$ne": True},
        }
    )
    if existing:
        raise APIError(
            "You are already subscribed to this trigger",
            status_code=409,
        )

    # Calculate next run time for the child
    next_run = None
    if parent.schedule:
        next_run = calculate_next_scheduled_run(parent.schedule)

    # Create child trigger (snapshot copy)
    child_trigger = Trigger(
        name=parent.name,
        prompt=parent.prompt,
        schedule=copy.deepcopy(parent.schedule) if parent.schedule else None,
        user=subscriber_id,
        agent=parent.agent,
        session=None,  # Created on first run
        parent_trigger=parent.id,
        subscribable=False,  # Children cannot be subscribed to
        status="active",
        next_scheduled_run=next_run,
    )
    child_trigger.save()

    logger.info(
        f"[SUBSCRIPTION] User {subscriber_user_id} subscribed to trigger {trigger_id}, "
        f"created child trigger {child_trigger.id}"
    )

    return child_trigger


async def unsubscribe_from_trigger(trigger_id: str, user_id: str) -> bool:
    """Unsubscribe a user from a trigger.

    Args:
        trigger_id: ID of the parent trigger
        user_id: ID of the user unsubscribing

    Returns:
        True if successfully unsubscribed, False if no subscription found
    """
    user_oid = ObjectId(user_id)

    # Find the child trigger for this user
    child = Trigger.find_one(
        {
            "parent_trigger": ObjectId(trigger_id),
            "user": user_oid,
            "deleted": {"$ne": True},
        }
    )

    if not child:
        return False

    # Soft delete and pause the child trigger
    child.update(deleted=True, status="paused")

    logger.info(
        f"[SUBSCRIPTION] User {user_id} unsubscribed from trigger {trigger_id}, "
        f"deleted child trigger {child.id}"
    )

    return True


async def _ensure_trigger_has_session(trigger: Trigger) -> Trigger:
    """Ensure trigger has a session. If not, create one.

    Args:
        trigger: The trigger to check

    Returns:
        Updated trigger with session set

    Raises:
        APIError: If trigger has no agent set or user lacks permissions
    """
    if trigger.session:
        return trigger

    # Trigger has no session - need to create one
    if not trigger.agent:
        raise APIError(
            "Trigger has no session and no agent specified. Cannot create session.",
            status_code=400,
        )

    # Load agent and check permissions
    agent = Agent.from_mongo(trigger.agent)
    if not agent:
        raise APIError(f"Agent {trigger.agent} not found", status_code=404)

    # Check if user has permissions for this agent
    # User must be either: 1) agent owner, or 2) have owner/editor permission

    # is_agent_owner = agent.owner == trigger.user
    # try:
    #     permission = AgentPermission.load(agent=trigger.agent, user=trigger.user)
    #     has_permission = permission and permission.level in [
    #         "owner",
    #         "editor",
    #         "member",
    #     ]
    # except Exception:
    #     has_permission = False

    # if not is_agent_owner and not has_permission:
    #     raise APIError(
    #         f"User does not have permission to use agent {agent.username}",
    #         status_code=403,
    #     )

    # Create new session
    new_session = Session(
        owner=trigger.user,
        agents=[trigger.agent],
        title=trigger.name,
        trigger=trigger.id,
        platform="app",
    )
    new_session.save()

    # Update trigger with new session
    trigger.update(session=new_session.id)
    logger.info(
        f"[TRIGGER] Created new session {new_session.id} for trigger {trigger.id}"
    )

    # Reload trigger to get updated session field
    return Trigger.from_mongo(trigger.id)


async def _load_trigger_dependencies(
    trigger_id: str,
) -> tuple[Trigger, Session, Agent, User]:
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

    # Ensure trigger has a session (creates one if needed)
    trigger = await _ensure_trigger_has_session(trigger)

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

    # For child triggers, check if parent is still active
    if trigger.parent_trigger:
        parent = Trigger.from_mongo(trigger.parent_trigger)
        if not parent or parent.deleted or parent.status == "paused":
            logger.info(
                f"[TRIGGER_ASYNC] Skipping child trigger {trigger_id} - parent is inactive/deleted"
            )
            trigger.update(status="paused")
            return None

    # For scheduled triggers, atomically set to running
    # For manual triggers (skip_message_add=True), status was already set by handle_trigger_run
    if not skip_message_add:
        trigger = atomic_set_running(trigger_id)
        if not trigger:
            logger.warning(
                "[TRIGGER_ASYNC] Duplicate execution prevented (already running)"
            )
            return None

    # Ensure trigger has a session (creates one if needed)
    trigger = await _ensure_trigger_has_session(trigger)

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
        # Reload trigger to get latest status (fix race condition)
        trigger = Trigger.from_mongo(trigger_id)
        if trigger and trigger.status == "running":
            # If no schedule, task stays active (manual-only) - clear next_scheduled_run
            if not trigger.schedule:
                trigger.update(status="active", next_scheduled_run=None)
            else:
                # Wrap in try/catch to prevent unhandled exceptions from leaving trigger stuck
                try:
                    next_run = calculate_next_scheduled_run(trigger.schedule)
                    if next_run:
                        trigger.update(status="active", next_scheduled_run=next_run)
                        logger.info(f"[TRIGGER_ASYNC] Next run scheduled: {next_run}")
                    else:
                        trigger.update(status="finished", next_scheduled_run=None)
                        logger.info(
                            "[TRIGGER_ASYNC] Trigger finished (no more scheduled runs)"
                        )
                except Exception as e:
                    logger.error(f"[TRIGGER_ASYNC] Failed to calculate next run: {e}")
                    # Clear next_scheduled_run to prevent phantom runs, keep status active
                    trigger.update(status="active", next_scheduled_run=None)
                    sentry_sdk.capture_exception(e)

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
        trigger, session, agent, user = await _load_trigger_dependencies(trigger_id)
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
