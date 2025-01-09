import logging
import os
from typing import Optional
import aiohttp
from bson import ObjectId
from fastapi import BackgroundTasks
from ably import AblyRest
from apscheduler.schedulers.background import BackgroundScheduler
import traceback
import asyncio
from contextlib import asynccontextmanager

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
    update_config: UpdateConfig, ably_client: AblyRest
) -> Optional[AblyRest]:
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


# Connection pool with TTL
class AblyConnectionPool:
    def __init__(self, client: AblyRest, ttl_seconds: int = 30):
        self.client = client
        self.ttl = ttl_seconds
        self.last_used = 0
        self._cleanup_task = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def get_connection(self):
        async with self._lock:
            if not self.client.connection.state == "connected":
                self.client.connect()
                while not self.client.connection.state == "connected":
                    await asyncio.sleep(0.1)

                if not self._cleanup_task:
                    self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.last_used = asyncio.get_event_loop().time()

        try:
            yield self.client
        except Exception as e:
            logger.error(f"Error with Ably connection: {str(e)}")
            await self.client.close()
            raise

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_used > self.ttl:
                async with self._lock:
                    if current_time - self.last_used > self.ttl:
                        await self.client.close()
                        self._cleanup_task = None
                        break


async def emit_update(update_config: UpdateConfig, data: dict):
    if not update_config:
        return

    if update_config.update_endpoint:
        await emit_http_update(update_config, data)
    elif update_config.sub_channel_name:
        try:
            client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = client.channels.get(update_config.sub_channel_name)
            await channel.publish(update_config.sub_channel_name, data)
        except Exception as e:
            logger.error(f"Failed to publish to Ably: {str(e)}")


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


async def load_existing_triggers(
    scheduler: BackgroundScheduler, ably_client: AblyRest, handle_chat_fn
):
    """Load all existing triggers from the database and add them to the scheduler"""
    from ..trigger import create_chat_trigger

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
