import logging
import os
from typing import Optional
import aiohttp
from bson import ObjectId
from fastapi import BackgroundTasks
from ably import AblyRest
import traceback
import re

from eve import deploy, trigger
from eve.api.errors import APIError
from eve.deploy import (
    authenticate_modal_key,
    check_environment_exists,
    create_environment,
)
from eve.tool import Tool
from eve.user import User
from eve.agent.agent import Agent
from eve.agent.thread import Thread
from eve.agent.tasks import async_title_thread
from eve.api.api_requests import ChatRequest, UpdateConfig
from eve.deploy import Deployment

logger = logging.getLogger(__name__)


async def get_update_channel(
    update_config: UpdateConfig, ably_client: AblyRest
) -> Optional[AblyRest]:
    return ably_client.channels.get(str(update_config.sub_channel_name))


async def setup_chat(
    request: ChatRequest,
    cache: bool = False,
    background_tasks: BackgroundTasks = None,
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

    tools = agent.get_tools(cache=cache)

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
    print("EMIT UPDATE CONFIG:", update_config)
    print("EMIT UPDATE DATA:", data)
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


async def create_telegram_chat_request(
    update_data: dict, deployment: Deployment
) -> Optional[ChatRequest]:
    message = update_data.get("message", {})
    if not message:
        return None

    chat_id = message.get("chat", {}).get("id")
    message_thread_id = message.get("message_thread_id")

    # Check allowlist if it exists
    if deployment.config and deployment.config.telegram:
        allowlist = deployment.config.telegram.topic_allowlist or []
        if allowlist:
            current_id = (
                f"{chat_id}_{message_thread_id}" if message_thread_id else str(chat_id)
            )
            if not any(item.id == current_id for item in allowlist):
                return None

    agent = Agent.from_mongo(deployment.agent)

    # Get user info
    from_user = message.get("from", {})
    user_id = str(from_user.get("id"))
    username = from_user.get("username", "unknown")
    user = User.from_telegram(user_id, username)

    # Process text and attachments
    text = message.get("text", "")
    attachments = []

    # Handle photos
    photos = message.get("photo", [])
    if photos:
        # Get the largest photo (last in array)
        largest_photo = photos[-1]
        file_id = largest_photo.get("file_id")

        # Initialize bot to get file path
        from telegram import Bot

        bot = Bot(deployment.secrets.telegram.token)
        file = await bot.get_file(file_id)
        photo_url = file.file_path
        attachments.append(photo_url)

        # Use caption as text if available
        if message.get("caption"):
            text = message.get("caption")

    # Create thread
    thread_key = (
        f"telegram-{chat_id}-topic-{message_thread_id}"
        if message_thread_id
        else f"telegram-{chat_id}"
    )
    thread = agent.request_thread(key=thread_key)

    # Clean message text (remove bot mention)
    cleaned_text = text
    if text:
        bot_username = f"@{agent.username.lower()}_bot"
        pattern = rf"\s*{re.escape(bot_username)}\b"
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return {
        "user_id": str(user.id),
        "agent_id": str(deployment.agent),
        "thread_id": str(thread.id),
        "user_is_bot": from_user.get("is_bot", False),
        "force_reply": True,
        "user_message": {
            "content": cleaned_text,
            "name": username,
            "attachments": attachments,
        },
        "update_config": {
            "update_endpoint": f"{os.getenv('EDEN_API_URL')}/emissions/platform/telegram",
            "deployment_id": str(deployment.id),
            "telegram_chat_id": str(chat_id),
            "telegram_message_id": str(message.get("message_id")),
            "telegram_thread_id": str(message_thread_id) if message_thread_id else None,
        },
    }
