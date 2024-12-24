import os
import json
import modal
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel
from typing import Optional

from eve.app.auth import auth
from eve.tool import Tool, get_tools_from_mongo
from eve.llm import UpdateType, UserMessage, async_prompt_thread
from eve.thread import Thread
from eve.app.database.mongo import serialize_document
from eve.agent import Agent
from eve.app.schemas.user import User

from eve.app.routers.taskRoutesV2 import router as Create
from eve.app.routers.creatorRoutesV2 import router as GetCreatorMe


# Config setup
db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
app_name = "tools-new" if db == "PROD" else "tools-new-dev"

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
background_tasks: BackgroundTasks = BackgroundTasks()

# FastAPI setup
web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# web_app.post("/create")(task_handler)
# web_app.post("/chat")(chat_handler)
# web_app.post("/chat/stream")(chat_stream)

web_app.include_router(Create, tags=["Create"])
web_app.include_router(GetCreatorMe, tags=["CreatorMe"])
    
# @web_app.post("/create")
# async def task(request: TaskRequest): #, auth: dict = Depends(auth.authenticate)):
#     return await handle_task(request.tool, auth.userId, request.args)


class ChatRequest(BaseModel):
    user_id: str
    agent_id: str
    thread_id: Optional[str] = None
    user_message: dict

@web_app.post("/chat")
async def handle_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin)
):
    user_id = request.user_id
    agent_id = request.agent_id
    thread_id = request.thread_id
    user_message = UserMessage(**request.user_message)

    tools = get_tools_from_mongo(db=db)
    
    user = User.from_mongo(str(user_id), db=db)
    agent = Agent.from_mongo(str(agent_id), db=db)

    if not thread_id:
        print("creating new thread")
        thread = agent.request_thread(db=db, user=user.id)
        print("new thread", thread.id)
    else:
        print("loading thread from thread_id", thread_id)
        thread = Thread.from_mongo(str(thread_id), db=db)
        print("got thread", thread.id)

    try:
        async def run_prompt():
            async for _ in async_prompt_thread(
                db=db,
                user=user,
                agent=agent,
                thread=thread,
                user_messages=user_message,
                tools=tools,
                force_reply=True,
                model="claude-3-5-sonnet-20241022",
            ):
                pass
        
        background_tasks.add_task(run_prompt)

        return {"status": "success", "thread_id": str(thread.id)}
    
    except Exception as e:
        print(e)
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
