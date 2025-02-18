import logging
import os
import subprocess
from typing import Dict, Any
from bson import ObjectId
import modal
import modal.runner

from eve.api.api_requests import CronSchedule
from eve.mongo import Collection, Document
from eve.services.trigger_fn import trigger_app, trigger_fn, base_image

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()
TRIGGER_ENV_NAME = "triggers"


@Collection("triggers")
class Trigger(Document):
    trigger_id: str
    user: ObjectId
    agent: ObjectId
    thread: ObjectId
    schedule: Dict[str, Any]
    message: str
    update_config: Dict[str, Any]

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        if isinstance(data.get("thread"), str):
            data["thread"] = ObjectId(data["thread"])
        super().__init__(**data)


async def create_chat_trigger(
    schedule: CronSchedule,
    trigger_id: str,
) -> None:
    """Creates a Modal scheduled function with the provided cron schedule"""
    try:
        # Convert schedule to cron string
        schedule_dict = schedule
        cron_string = f"{schedule_dict.get('minute', '*')} {schedule_dict.get('hour', '*')} {schedule_dict.get('day', '*')} {schedule_dict.get('month', '*')} {schedule_dict.get('day_of_week', '*')}"
        # Apply schedule and deploy exactly like the example
        trigger_app.function(
            schedule=modal.Cron(cron_string),
            image=base_image,
            secrets=[
                modal.Secret.from_name("eve-secrets", environment_name="main"),
                modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
            ],
        )(trigger_fn)
        modal.runner.deploy_app(
            trigger_app, name=f"{trigger_id}", environment_name=TRIGGER_ENV_NAME
        )

        logger.info(f"Created Modal trigger {trigger_id} with schedule: {cron_string}")

    except Exception as e:
        error_msg = f"Failed to create Modal trigger: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


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
