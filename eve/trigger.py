from typing import Dict, Any
from bson import ObjectId
from eve.mongo import Collection, Document
import asyncio
import logging
from typing import Optional
from fastapi import BackgroundTasks
from ably import AblyRealtime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import sentry_sdk

from eve.api.api_requests import ChatRequest, UpdateConfig
from eve.thread import UserMessage

logger = logging.getLogger(__name__)


@Collection("triggers")
class Trigger(Document):
    trigger_id: str
    user: ObjectId
    agent: ObjectId
    schedule: Dict[str, Any]
    message: str
    update_config: Dict[str, Any]

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        super().__init__(**data)


async def create_chat_trigger(
    user_id: str,
    agent_id: str,
    message: str,
    schedule: dict,
    update_config: Optional[UpdateConfig],
    scheduler: BackgroundScheduler,
    ably_client: AblyRealtime,
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
            chat_request = ChatRequest(
                user_id=user_id,
                agent_id=agent_id,
                user_message=UserMessage(content=message),
                update_config=update_config,
                force_reply=True,
            )

            result = loop.run_until_complete(
                handle_chat_fn(
                    request=chat_request,
                    background_tasks=background_tasks,
                    ably_client=ably_client,
                )
            )
            loop.run_until_complete(background_tasks())
            logger.info(f"Completed scheduled chat for trigger {trigger_id}: {result}")

        except Exception as e:
            error_msg = "Sorry, there was an error in your scheduled chat."
            logger.error(error_msg, exc_info=True)
            sentry_sdk.capture_exception(e)
            try:
                if update_config and update_config.sub_channel_name:
                    channel = ably_client.channels.get(
                        str(update_config.sub_channel_name)
                    )
                    loop.run_until_complete(
                        channel.publish("update", {"type": "error", "error": error_msg})
                    )
            except Exception as notify_error:
                logger.error(
                    f"Failed to notify about trigger error: {str(notify_error)}",
                    exc_info=True,
                )
                sentry_sdk.capture_exception(notify_error)
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
