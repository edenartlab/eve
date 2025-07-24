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




@handle_errors
async def create_trigger_fn(
    schedule: CronSchedule,
    trigger_id: str,
) -> datetime:
    """Calculate and return the next scheduled run time"""
    print(f"Creating session trigger {trigger_id} with schedule {schedule}")
    schedule_dict = schedule
    
    # Calculate next scheduled run
    next_run = calculate_next_scheduled_run(schedule_dict)
    
    if next_run:
        logger.info(f"Trigger {trigger_id} next scheduled run: {next_run}")
    else:
        logger.warning(f"Could not calculate next run time for trigger {trigger_id}")
    
    return next_run


async def stop_trigger(trigger_id: str) -> None:
    """Mark trigger as stopped in database"""
    logger.info(f"Stopping trigger {trigger_id}")
    # No need to stop Modal app anymore since we're using a centralized scheduler
