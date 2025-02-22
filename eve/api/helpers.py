import logging
import os
import time
from typing import Optional
import aiohttp
from bson import ObjectId
from fastapi import BackgroundTasks
from ably import AblyRest
import traceback

from eve import deploy, trigger
from eve.api.errors import APIError
from eve.deploy import (
    authenticate_modal_key,
    check_environment_exists,
    create_environment,
)
from eve.tool import Tool
from eve.user import User
from eve.agent import Agent
from eve.thread import Thread
from eve.llm import async_title_thread
from eve.api.api_requests import ChatRequest, UpdateConfig

logger = logging.getLogger(__name__)


async def get_update_channel(
    update_config: UpdateConfig, ably_client: AblyRest
) -> Optional[AblyRest]:
    return ably_client.channels.get(str(update_config.sub_channel_name))


async def setup_chat(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> tuple[User, Agent, Thread, list[Tool]]:
    try:
        user = User.from_mongo(request.user_id)
    except Exception as e:
        logger.error(f"Error loading user: {traceback.format_exc()}")
        raise APIError(f"Invalid user_id: {request.user_id}", status_code=400) from e

    try:
        agent = Agent.from_mongo(request.agent_id, cache=False)
    except Exception as e:
        logger.error(f"Error loading agent: {traceback.format_exc()}")
        raise APIError(f"Invalid agent_id: {request.agent_id}", status_code=400) from e

    tools = agent.get_tools(cache=False)

    if request.thread_id:
        try:
            thread = Thread.from_mongo(request.thread_id)
        except Exception as e:
            logger.error(f"Error loading thread: {traceback.format_exc()}")
            raise APIError(
                f"Invalid thread_id: {request.thread_id}", status_code=400
            ) from e
    else:
        thread = agent.request_thread(user=user.id, message_limit=25)
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


async def emit_update(update_config: Optional[UpdateConfig], data: dict):
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


def pre_modal_setup():
    authenticate_modal_key()
    if not check_environment_exists(deploy.DEPLOYMENT_ENV_NAME):
        create_environment(deploy.DEPLOYMENT_ENV_NAME)
    if not check_environment_exists(trigger.TRIGGER_ENV_NAME):
        create_environment(trigger.TRIGGER_ENV_NAME)
