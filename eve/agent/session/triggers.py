import logging
import os
import pytz
from datetime import datetime, timezone
from apscheduler.triggers.cron import CronTrigger

from eve.api.api_requests import CronSchedule
from eve.api.errors import handle_errors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()


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
        end_date=end_date
    )
    
    # Get next fire time
    next_time = trigger.get_next_fire_time(None, datetime.now(pytz.timezone(timezone_str)))
    if next_time:
        # Convert to UTC for storage
        return next_time.astimezone(pytz.UTC).replace(tzinfo=timezone.utc)
    return None


# @handle_errors
# async def create_trigger_fn(
#     schedule: CronSchedule,
#     trigger_id: str,
# ) -> datetime:
#     """Calculate and return the next scheduled run time"""
#     print(f"Creating session trigger {trigger_id} with schedule {schedule}")
#     schedule_dict = schedule
    
#     # Calculate next scheduled run
#     next_run = calculate_next_scheduled_run(schedule_dict)
    
#     if next_run:
#         logger.info(f"Trigger {trigger_id} next scheduled run: {next_run}")
#     else:
#         logger.warning(f"Could not calculate next run time for trigger {trigger_id}")
    
#     return next_run


# async def execute_trigger(trigger, is_immediate: bool = False):
#     """
#     Execute a trigger using the current session-based system.
#     Used by both scheduled and immediate trigger execution.
    
#     Args:
#         trigger: Trigger model instance
#         is_immediate: Whether this is an immediate execution (affects notifications)
    
#     Returns:
#         dict: Response from the session prompt endpoint
#     """
#     import aiohttp
    
#     logger.info(f"Executing trigger {trigger.trigger_id} (immediate={is_immediate})")
    
#     # Prepare the prompt session request
#     request_data = {
#         "session_id": str(trigger.session) if trigger.session else None,
#         "user_id": str(trigger.user),
#         "actor_agent_ids": [str(trigger.agent)],
#         "message": {
#             "role": "system",
#             "content": f"""## Task

# You have been given the following instructions. Do not ask for clarification, or stop until you have completed the task.

# {trigger.instruction}

# """,
#         },
#         "update_config": trigger.update_config,
#     }
    
#     # If no session, add creation args
#     if not trigger.session:
#         request_data["creation_args"] = {
#             "owner_id": str(trigger.user),
#             "agents": [str(trigger.agent)],
#             "trigger": str(trigger.id),
#         }
    
#     # Add notification configuration
#     notification_type = "Immediate Task" if is_immediate else "Task"
#     request_data["notification_config"] = {
#         "user_id": str(trigger.user),
#         "notification_type": "trigger_complete",
#         "title": f"{notification_type} Completed Successfully",
#         "message": f"Your {'immediate ' if is_immediate else 'scheduled '}task has completed successfully: {trigger.instruction[:100]}...",
#         "trigger_id": str(trigger.id),
#         "agent_id": str(trigger.agent),
#         "priority": "normal",
#         "metadata": {
#             "trigger_id": trigger.trigger_id,
#             "immediate_run": is_immediate,
#         },
#         "success_notification": True,
#         "failure_notification": True,
#         "failure_title": f"{notification_type} Failed",
#         "failure_message": f"Your {'immediate ' if is_immediate else 'scheduled '}task failed: {trigger.instruction[:100]}...",
#     }
    
#     # Make async HTTP POST to prompt session endpoint
#     api_url = os.getenv("EDEN_API_URL")
#     async with aiohttp.ClientSession() as session:
#         async with session.post(
#             f"{api_url}/sessions/prompt",
#             json=request_data,
#             headers={
#                 "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
#                 "Content-Type": "application/json",
#             },
#         ) as response:
#             if response.status != 200:
#                 error_text = await response.text()
#                 logger.error(
#                     f"Failed to execute trigger {trigger.trigger_id}: {error_text}"
#                 )
#                 raise Exception(f"Failed to execute trigger: {response.status} - {error_text}")
            
#             result = await response.json()
#             session_id = result.get("session_id")
            
#             logger.info(f"Successfully executed trigger {trigger.trigger_id}, session: {session_id}")
#             return result


# async def stop_trigger(trigger_id: str) -> None:
#     """Mark trigger as stopped in database"""
#     logger.info(f"Stopping trigger {trigger_id}")
#     # No need to stop Modal app anymore since we're using a centralized scheduler
