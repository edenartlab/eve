import logging
from typing import Optional
from bson import ObjectId
from fastapi import BackgroundTasks, FastAPI
from ably import AblyRealtime

from eve.tool import Tool
from eve.user import User
from eve.agent import Agent
from eve.thread import Thread
from eve.llm import async_title_thread
from eve.api.requests import ChatRequest

logger = logging.getLogger(__name__)


async def setup_chat(
    web_app: FastAPI, request: ChatRequest, background_tasks: BackgroundTasks
) -> tuple[User, Agent, Thread, list[Tool], Optional[AblyRealtime]]:
    update_channel = None
    if request.update_config and request.update_config.sub_channel_name:
        try:
            update_channel = web_app.state.ably_client.channels.get(
                str(request.update_config.sub_channel_name)
            )
        except Exception as e:
            logger.error(f"Failed to create Ably channel: {str(e)}")

    user = User.from_mongo(request.user_id)
    agent = Agent.from_mongo(request.agent_id, cache=True)
    tools = agent.get_tools(cache=True)

    if request.thread_id:
        thread = Thread.from_mongo(request.thread_id)
    else:
        thread = agent.request_thread(user=user.id)
        background_tasks.add_task(async_title_thread, thread, request.user_message)

    return user, agent, thread, tools, update_channel

def serialize_for_json(obj):
    """Recursively serialize objects for JSON, handling ObjectId and other special types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


async def handle_task(tool: str, user_id: str, args: dict = {}) -> dict:
    tool = Tool.load(key=tool)
    return await tool.async_start_task(requester_id=user_id, user_id=user_id, args=args)
