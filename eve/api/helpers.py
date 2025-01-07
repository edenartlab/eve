import logging
import os
from typing import Optional
import aiohttp
from bson import ObjectId
from fastapi import BackgroundTasks
from ably import AblyRealtime
from apscheduler.schedulers.background import BackgroundScheduler
import traceback

from eve.tool import Tool
from eve.user import User
from eve.agent import Agent
from eve.thread import Thread
from eve.llm import async_title_thread
from eve.api.api_requests import ChatRequest, UpdateConfig
from eve.trigger import Trigger
from eve.mongo import get_collection

logger = logging.getLogger(__name__)


async def get_update_channel(
    update_config: UpdateConfig, ably_client: AblyRealtime
) -> Optional[AblyRealtime]:
    return ably_client.channels.get(str(update_config.sub_channel_name))


async def setup_chat(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> tuple[User, Agent, Thread, list[Tool]]:
    user = User.from_mongo(request.user_id)
    agent = Agent.from_mongo(request.agent_id, cache=True)
    tools = agent.get_tools(cache=True)

    if request.thread_id:
        thread = Thread.from_mongo(request.thread_id)
    else:
        thread = agent.request_thread(user=user.id)
        background_tasks.add_task(async_title_thread, thread, request.user_message)

    return user, agent, thread, tools


def serialize_for_json(obj):
    """Recursively serialize objects for JSON, handling ObjectId and other special types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


async def emit_update(
    update_config: UpdateConfig, update_channel: AblyRealtime, data: dict
):
    if update_config and update_config.update_endpoint:
        raise ValueError("update_endpoint and sub_channel_name cannot be used together")
    elif update_config.update_endpoint:
        await emit_http_update(update_config, data)
    elif update_config.sub_channel_name:
        await emit_channel_update(update_channel, data)
    else:
        raise ValueError("One of update_endpoint or sub_channel_name must be provided")


async def emit_http_update(update_config: UpdateConfig, data: dict):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                update_config.update_endpoint,
                json=data,
                headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to send update to endpoint: {await response.text()}"
                    )
        except Exception as e:
            logger.error(f"Error sending update to endpoint: {str(e)}")


async def emit_channel_update(update_channel: AblyRealtime, data: dict):
    try:
        await update_channel.publish("update", data)
    except Exception as e:
        logger.error(f"Failed to publish to Ably: {str(e)}")


async def load_existing_triggers(
    scheduler: BackgroundScheduler, ably_client: AblyRealtime, handle_chat_fn
):
    """Load all existing triggers from the database and add them to the scheduler"""
    from eve.trigger import create_chat_trigger

    triggers_collection = get_collection(Trigger.collection_name)

    for trigger_doc in triggers_collection.find({}):
        try:
            # Convert mongo doc to Trigger object
            trigger = Trigger.convert_from_mongo(trigger_doc)
            trigger = Trigger.from_schema(trigger)

            await create_chat_trigger(
                user_id=str(trigger.user),
                agent_id=str(trigger.agent),
                message=trigger.message,
                schedule=trigger.schedule,
                update_config=UpdateConfig(**trigger.update_config)
                if trigger.update_config
                else None,
                scheduler=scheduler,
                ably_client=ably_client,
                trigger_id=trigger.trigger_id,
                handle_chat_fn=handle_chat_fn,
            )
            logger.info(f"Loaded trigger {trigger.trigger_id}")

        except Exception as e:
            logger.error(
                f"Error loading trigger {trigger_doc.get('trigger_id', 'unknown')}: {str(e)}\n{traceback.format_exc()}"
            )
