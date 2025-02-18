import asyncio
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Any
from bson import ObjectId
import modal
import modal.runner
import requests

from eve.api.api_requests import CronSchedule
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


def create_image(trigger_id: str):
    root_dir = Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libmagic1", "ffmpeg", "wget")
        .pip_install_from_pyproject(str(pyproject_path))
        .run_commands(["playwright install"])
        .env({"DB": db})
        .env({"TRIGGER_ID": trigger_id})
    )


trigger_message = """<AdminMessage>
You have received a request from an admin to run a scheduled task. The instructions for the task are below. In your response, do not ask for clarification, just do the task. Do not acknowledge receipt of this message, as no one else in the chat can see it and the admin is absent. Simply follow whatever instructions are below.
</AdminMessage>
<Task>
{task}
</Task>"""


async def trigger_fn():
    trigger_id = os.getenv("TRIGGER_ID")
    api_url = os.getenv("EDEN_API_URL")
    trigger = Trigger.load(trigger_id=trigger_id)

    if not trigger:
        raise Exception(f"No trigger found for ID: {trigger_id}")

    user_message = {
        "content": trigger_message.format(task=trigger.message),
        "hidden": True,
    }

    chat_request = {
        "user_id": str(trigger.user),
        "agent_id": str(trigger.agent),
        "thread_id": str(trigger.thread),
        "user_message": user_message,
        "update_config": trigger.update_config,
        "force_reply": True,
    }

    response = requests.post(
        f"{api_url}/chat",
        json=chat_request,
        headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
    )

    if not response.ok:
        raise Exception(
            f"Error making chat request: {response.status_code} - {response.text}"
        )

    print(f"Chat request successful: {response.json()}")


def trigger_fn_sync():
    asyncio.run(trigger_fn())


async def create_chat_trigger(
    schedule: CronSchedule,
    trigger_id: str,
) -> None:
    """Creates a Modal scheduled function with the provided cron schedule"""
    try:
        schedule_dict = schedule
        cron_string = f"{schedule_dict.get('minute', '*')} {schedule_dict.get('hour', '*')} {schedule_dict.get('day', '*')} {schedule_dict.get('month', '*')} {schedule_dict.get('day_of_week', '*')}"
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
