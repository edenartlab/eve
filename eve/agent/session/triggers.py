import logging
import os
import subprocess
import pytz
from datetime import datetime
import modal
import modal.runner

from eve.agent.session.trigger_fn import trigger_fn
from eve.api.api_requests import CronSchedule
from eve.api.errors import handle_errors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = os.getenv("DB", "STAGE").upper()
TRIGGER_ENV_NAME = "triggers"

trigger_app = modal.App()


def create_image(trigger_id: str):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libmagic1", "ffmpeg", "wget")
        .pip_install_from_pyproject("/eve/pyproject.toml")
        .run_commands(["playwright install"])
        .env({"DB": db})
        .env({"TRIGGER_ID": trigger_id})
    )


@handle_errors
async def create_trigger_fn(
    schedule: CronSchedule,
    trigger_id: str,
) -> None:
    print(f"Creating session trigger {trigger_id} with schedule {schedule}")
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

        # Get day_of_month, fallback to day for backwards compatibility
        day_of_month = schedule_dict.get("day_of_month") or schedule_dict.get(
            "day", "*"
        )

        # Create cron string with potentially adjusted values
        cron_string = f"{minute} {hour} {day_of_month} {schedule_dict.get('month', '*')} {schedule_dict.get('day_of_week', '*')}"

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


async def stop_trigger(trigger_id: str) -> None:
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
