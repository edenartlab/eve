import asyncio
import logging
import sentry_sdk
from typing import Dict, Any
from bson import ObjectId
from typing import Optional
from fastapi import BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .api.api_requests import ChatRequest, UpdateConfig
from .thread import UserMessage
from .mongo import Collection, Document

logger = logging.getLogger(__name__)


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



trigger_message = """<AdminMessage>
You have received a request from an admin to run a scheduled task. The instructions for the task are below. In your response, do not ask for clarification, just do the task. Do not acknowledge receipt of this message, as no one else in the chat can see it and the admin is absent. Simply follow whatever instructions are below.
</AdminMessage>
<Task>
{task}
</Task>"""


async def create_chat_trigger(
    user_id: str,
    agent_id: str,
    thread_id: str,
    message: str,
    schedule: dict,
    update_config: Optional[UpdateConfig],
    scheduler: BackgroundScheduler,
    trigger_id: str,
    handle_chat_fn,
):
    """Creates and adds a scheduled chat job to the scheduler"""

    def run_scheduled_task():
        logger.info(f"Running scheduled chat for user {user_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            background_tasks = BackgroundTasks()
            
            user_message = UserMessage(
                content=trigger_message.format(task=message), 
                hidden=True
            )

            chat_request = ChatRequest(
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                user_message=user_message,
                update_config=update_config,
                force_reply=True,
            )

            result = loop.run_until_complete(
                handle_chat_fn(
                    request=chat_request,
                    background_tasks=background_tasks,
                )
            )
            loop.run_until_complete(background_tasks())
            logger.info(f"Completed scheduled chat for trigger {trigger_id}: {result}")

        except Exception as e:
            error_msg = "Sorry, there was an error in your scheduled chat."
            logger.error(error_msg, exc_info=True)
            sentry_sdk.capture_exception(e)
        finally:
            loop.close()

    try:
        job = scheduler.add_job(
            run_scheduled_task,
            trigger=CronTrigger(**schedule),
            id=trigger_id,
            misfire_grace_time=None,
            coalesce=True,
        )
        return job

    except Exception as e:
        error_msg = f"Failed to create trigger job {trigger_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        sentry_sdk.capture_exception(e)
        raise


async def load_existing_triggers(scheduler: BackgroundScheduler, handle_chat_fn):
    """Loads all existing triggers from the database into the scheduler"""
    collection = Trigger.get_collection()
    triggers = collection.find({})

    for trigger in triggers:
        try:
            await create_chat_trigger(
                user_id=str(trigger["user"]),
                agent_id=str(trigger["agent"]),
                thread_id=str(trigger["thread"]),
                message=trigger["message"],
                schedule=trigger["schedule"],
                update_config=trigger["update_config"],
                scheduler=scheduler,
                trigger_id=trigger["trigger_id"],
                handle_chat_fn=handle_chat_fn,
            )
            logger.info(f"Loaded trigger {trigger['trigger_id']}")
        except Exception as e:
            logger.error(
                f"Failed to load trigger {trigger.get('trigger_id', 'unknown')}: {str(e)}"
            )
            sentry_sdk.capture_exception(e)
