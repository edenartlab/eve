import os
import json
import modal
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel, ConfigDict
from typing import Optional
from bson import ObjectId
import logging
from ably import AblyRealtime
from pathlib import Path
import aiohttp
import traceback

from eve import auth
from eve.tool import Tool
from eve.llm import UpdateType, UserMessage, async_prompt_thread, async_title_thread
from eve.thread import Thread
from eve.mongo import serialize_document
from eve.agent import Agent
from eve.user import User

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

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
background_tasks: BackgroundTasks = BackgroundTasks()


# Store ably client at app state level
@web_app.on_event("startup")
async def startup_event():
    web_app.state.ably_client = AblyRealtime(os.getenv("ABLY_PUBLISHER_KEY"))


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
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
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
    force_reply: bool = False


def serialize_for_json(obj):
    """Recursively serialize objects for JSON, handling ObjectId and other special types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


async def setup_chat(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> tuple[User, Agent, Thread, list[Tool], Optional[AblyRealtime]]:
    update_channel = None
    if request.update_config and request.update_config.sub_channel_name:
        try:
            update_channel = web_app.state.ably_client.channels.get(
                str(request.update_config.sub_channel_name)
            )
        except Exception as e:
            logger.error(f"Failed to create Ably channel: {str(e)}")

    user = User.from_mongo(request.user_id, db=db)
    agent = Agent.from_mongo(request.agent_id, db=db, cache=True)
    tools = agent.get_tools(db=db, cache=True)

    if request.thread_id:
        thread = Thread.from_mongo(request.thread_id, db=db)
    else:
        thread = agent.request_thread(db=db, user=user.id)
        background_tasks.add_task(async_title_thread, thread, request.user_message)

    return user, agent, thread, tools, update_channel


@web_app.post("/chat")
async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(auth.authenticate_admin),
):
    print("handle_chat", request)
    try:
        user, agent, thread, tools, update_channel = await setup_chat(
            request, background_tasks
        )

        async def run_prompt():
            async for update in async_prompt_thread(
                db=db,
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                model="claude-3-5-sonnet-20241022",
                stream=False,
            ):
                data = {
                    "type": update.type.value,
                    "update_config": request.update_config.model_dump()
                    if request.update_config
                    else {},
                }

                if update.type == UpdateType.ASSISTANT_MESSAGE:
                    data["content"] = update.message.content
                elif update.type == UpdateType.TOOL_COMPLETE:
                    data["tool"] = update.tool_name
                    data["result"] = serialize_for_json(update.result)
                elif update.type == UpdateType.ERROR:
                    data["error"] = update.error if hasattr(update, "error") else None

                if request.update_config:
                    if request.update_config.update_endpoint:
                        async with aiohttp.ClientSession() as session:
                            try:
                                async with session.post(
                                    request.update_config.update_endpoint,
                                    json=data,
                                    headers={
                                        "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"
                                    },
                                ) as response:
                                    if response.status != 200:
                                        logger.error(
                                            f"Failed to send update to endpoint: {await response.text()}"
                                        )
                            except Exception as e:
                                logger.error(
                                    f"Error sending update to endpoint: {str(e)}"
                                )

                    elif update_channel:
                        try:
                            await update_channel.publish("update", data)
                        except Exception as e:
                            logger.error(f"Failed to publish to Ably: {str(e)}")

        background_tasks.add_task(run_prompt)
        return {"status": "success", "thread_id": str(thread.id)}

    except Exception as e:
        logger.error(f"Error in handle_chat: {str(e)}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@web_app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    auth: dict = Depends(auth.authenticate_admin),
):
    try:
        user, agent, thread, tools, update_channel = await setup_chat(
            request, BackgroundTasks()
        )

        async def event_generator():
            async for update in async_prompt_thread(
                db=db,
                user=user,
                agent=agent,
                thread=thread,
                user_messages=request.user_message,
                tools=tools,
                force_reply=request.force_reply,
                model="claude-3-5-sonnet-20241022",
                stream=True,
            ):
                data = {"type": update.type}
                if update.type == UpdateType.ASSISTANT_TOKEN:
                    data["token"] = getattr(update, "token", "")
                elif update.type == UpdateType.ASSISTANT_MESSAGE:
                    data["content"] = update.message.content
                    if update.message.tool_calls:
                        data["tool_calls"] = [
                            serialize_for_json(t.model_dump())
                            for t in update.message.tool_calls
                        ]
                elif update.type == UpdateType.TOOL_COMPLETE:
                    data["tool"] = update.tool_name
                    data["result"] = serialize_for_json(update.result)
                elif update.type == UpdateType.ERROR:
                    data["error"] = update.error or "Unknown error occurred"

                yield f"data: {json.dumps({'event': 'update', 'data': data})}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


# Modal app setup
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
    ],
)

root_dir = Path(__file__).parent.parent
workflows_dir = root_dir / ".." / "workflows"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE")})
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .copy_local_dir(str(workflows_dir), "/workflows")
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
