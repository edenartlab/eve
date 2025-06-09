import logging
import os
import subprocess
import pytz
from datetime import datetime
from typing import Dict, Any, Literal, Optional
from bson import ObjectId
import modal
import modal.runner

from eve.api.api_requests import CronSchedule
from eve.api.errors import handle_errors
from eve.functions.fn_trigger import trigger_fn
from eve.mongo import Collection, Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()
TRIGGER_ENV_NAME = "triggers"

trigger_app = modal.App()


@Collection("triggers")
class Trigger(Document):
    trigger_id: str
    user: ObjectId
    agent: ObjectId
    thread: ObjectId
    platform: str
    channel: Optional[Dict[str, Any]]
    schedule: Dict[str, Any]
    message: str
    update_config: Optional[Dict[str, Any]]
    status: Optional[Literal["active", "paused", "finished"]] = "active"

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        if isinstance(data.get("thread"), str):
            data["thread"] = ObjectId(data["thread"])
        if data.get("channel") is None:
            data["channel"] = None
        if data.get("update_config") is None:
            data["update_config"] = None
        if data.get("status") is None:
            data["status"] = "active"
        super().__init__(**data)


def create_image(trigger_id: str):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libmagic1", "ffmpeg", "wget")
        .pip_install_from_pyproject("/eve/pyproject.toml")
        .run_commands(["playwright install"])
        .env({"DB": db})
        .env({"TRIGGER_ID": trigger_id})
        .add_local_dir("/root/eve", "/root/eve")
    )


@handle_errors
async def create_chat_trigger(
    schedule: CronSchedule,
    trigger_id: str,
) -> None:
    print(f"Creating chat trigger {trigger_id} with schedule {schedule}")
    """Creates a Modal scheduled function with the provided cron schedule"""
    with modal.enable_output():
        schedule_dict = schedule

        # Get hour and minute from schedule
        hour = schedule_dict.get("hour", "*")
        minute = schedule_dict.get("minute", "*")
        timezone_str = schedule_dict.get("timezone")

        # If we have specific hour/minute values and a timezone, convert to UTC
        if timezone_str and hour != "*" and minute != "*":
            try:
                # Convert to integers for calculation
                hour_int = int(hour)
                minute_int = int(minute)

                # Create a datetime object with the scheduled time in the specified timezone
                user_tz = pytz.timezone(timezone_str)
                now = datetime.now()
                local_dt = user_tz.localize(
                    datetime(now.year, now.month, now.day, hour_int, minute_int)
                )

                # Convert to UTC
                utc_dt = local_dt.astimezone(pytz.UTC)

                # Update hour and minute to UTC values
                hour = str(utc_dt.hour)
                minute = str(utc_dt.minute)

                logger.info(
                    f"Converted schedule from {timezone_str} to UTC: {hour_int}:{minute_int} -> {hour}:{minute}"
                )
            except Exception as e:
                logger.error(
                    f"Error converting timezone: {str(e)}. Using original values."
                )

        # Create cron string with potentially adjusted values
        cron_string = f"{minute} {hour} {schedule_dict.get('day', '*')} {schedule_dict.get('month', '*')} {schedule_dict.get('day_of_week', '*')}"

        trigger_app.function(
            schedule=modal.Cron(cron_string),
            image=create_image(trigger_id),
            secrets=[
                modal.Secret.from_name("eve-secrets", environment_name="main"),
                modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
            ],
        )(trigger_fn)
        modal.runner.deploy_app(
            trigger_app, name=f"{trigger_id}", environment_name=TRIGGER_ENV_NAME
        )

        timezone_info = f" (converted from {timezone_str})" if timezone_str else ""
        logger.info(
            f"Created Modal trigger {trigger_id} with schedule: {cron_string}{timezone_info}"
        )


async def delete_trigger(trigger_id: str) -> None:
    """Deletes a Modal scheduled function"""
    try:
        result = subprocess.run(
            ["modal", "app", "stop", f"{trigger_id}", "-e", TRIGGER_ENV_NAME],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Deleted Modal trigger {trigger_id}")
        if result.returncode != 0:
            raise Exception(f"Failed to stop trigger: {result.stderr}")

    except Exception as e:
        error_msg = f"Failed to delete Modal trigger: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
