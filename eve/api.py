import os
import json
import modal
import asyncio
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel, ConfigDict
from typing import Optional
from bson import ObjectId
import logging

from eve import auth
from eve.tool import Tool, get_tools_from_mongo
from eve.llm import UpdateType, UserMessage, async_prompt_thread, async_title_thread
from eve.thread import Thread
from eve.mongo import serialize_document
from eve.agent import Agent
from eve.user import User
from ably import AblyRealtime

# Config and logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
app_name = "tools-new" if db == "PROD" else "tools-new-dev"

# FastAPI setup
web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ably_client = AblyRealtime(os.getenv("ABLY_PUBLISHER_KEY"))

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
background_tasks: BackgroundTasks = BackgroundTasks()

# web_app.post("/create")(task_handler)
# web_app.post("/chat")(chat_handler)
# web_app.post("/chat/stream")(chat_stream)


class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str


async def handle_task(tool: str, user_id: str, args: dict = {}) -> dict:
    tool = Tool.load(key=tool, db=db)
    return await tool.async_start_task(
        requester_id=user_id, user_id=user_id, args=args, db=db
    )


@web_app.post("/create")
async def task_admin(request: TaskRequest, _: dict = Depends(auth.authenticate_admin)):
    result = await handle_task(request.tool, request.user_id, request.args)
    return serialize_document(result.model_dump())


# @web_app.post("/create")
# async def task(request: TaskRequest): #, auth: dict = Depends(auth.authenticate)):
#     return await handle_task(request.tool, auth.userId, request.args)


class UpdateConfig(BaseModel):
    sub_channel_name: str
    discord_channel_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    cast_hash: Optional[str] = None
    author_fid: Optional[int] = None
    message_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatRequest(BaseModel):
    user_id: str
    agent_id: str
    user_message: UserMessage
    thread_id: Optional[str] = None
    update_config: Optional[UpdateConfig] = None


def serialize_for_json(obj):
    """Recursively serialize objects for JSON, handling ObjectId and other special types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


async def fetch_resources(
    user_id: str, agent_id: str, thread_id: Optional[str], db: str
):
    """Fetch user, agent, thread and tools in parallel"""
    user_task = asyncio.create_task(
        asyncio.to_thread(User.from_mongo, str(user_id), db)
    )
    agent_task = asyncio.create_task(
        asyncio.to_thread(Agent.from_mongo, str(agent_id), db)
    )
    tools_task = asyncio.create_task(asyncio.to_thread(get_tools_from_mongo, db))

    if thread_id:
        thread_task = asyncio.create_task(
            asyncio.to_thread(Thread.from_mongo, str(thread_id), db)
        )
    else:
        thread_task = None

    user = await user_task
    agent = await agent_task
    tools = await tools_task

    if thread_task:
        thread = await thread_task
    else:
        thread = agent.request_thread(db=db, user=user.id)

    return user, agent, thread, tools


@web_app.post("/chat")
async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    update_channel = None
    if request.update_config:
        update_channel = ably_client.channels.get(
            request.update_config.sub_channel_name
        )

    user, agent, thread, tools = await fetch_resources(
        request.user_id, request.agent_id, request.thread_id, db
    )

    try:

        async def run_prompt():
            async for update in async_prompt_thread(
                db=db,
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=True,
                model="claude-3-5-sonnet-20241022",
            ):
                if update_channel:
                    data = {
                        "type": update.type.value,
                        "update_config": request.update_config.model_dump(),
                    }

                    if update.type == UpdateType.ASSISTANT_MESSAGE:
                        data["content"] = update.message.content
                    elif update.type == UpdateType.TOOL_COMPLETE:
                        data["tool"] = update.tool_name
                        data["result"] = serialize_for_json(update.result)
                    elif update.type == UpdateType.ERROR:
                        data["error"] = (
                            update.error if hasattr(update, "error") else None
                        )

                    await update_channel.publish("update", data)

        background_tasks.add_task(run_prompt)
        return {"status": "success", "thread_id": str(thread.id)}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@web_app.post("/chat_and_wait")
async def stream_chat(
    request: ChatRequest,
    auth: dict = Depends(auth.authenticate),
):
    user_messages = UserMessage(**request.user_message)

    async def event_generator():
        async for update in async_prompt_thread(
            db=db,
            user_id=auth.userId,
            thread_name=request.thread_name,
            user_messages=user_messages,
            tools=get_tools_from_mongo(db=db),
            provider="anthropic",
        ):
            if update.type == UpdateType.ASSISTANT_MESSAGE:
                data = {
                    "type": str(update.type),
                    "content": update.message.content,
                }
            elif update.type == UpdateType.TOOL_COMPLETE:
                data = {
                    "type": str(update.type),
                    "tool": update.tool_name,
                    "result": update.result,
                }
            else:
                data = {
                    "type": "error",
                    "error": update.error or "Unknown error occurred",
                }

            yield f"data: {json.dumps({'event': 'update', 'data': data})}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Modal app setup
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name(s)
        for s in [
            "admin-key",
            "s3-credentials",
            "mongo-credentials",
            "gcp-credentials",
            "replicate",
            "openai",
            "anthropic",
            "elevenlabs",
            "hedra",
            "newsapi",
            "runway",
            "sentry",
        ]
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE")})
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("../pyproject.toml")
    .copy_local_dir("../../workflows", "/workflows")
)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=10,
    container_idle_timeout=60,
    timeout=3600,
)
@modal.asgi_app()
def fastapi_app():
    return web_app
