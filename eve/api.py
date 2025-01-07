import logging
import os
import threading
import modal
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from ably import AblyRealtime
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
from contextlib import asynccontextmanager

from eve import auth
from eve.api.handlers import (
    handle_cancel,
    handle_chat,
    handle_create,
    handle_deployment,
    handle_schedule,
    handle_stream_chat,
)
from eve.api.requests import (
    CancelRequest,
    ChatRequest,
    ScheduleRequest,
    TaskRequest,
    DeployRequest,
)
from eve.deploy import (
    authenticate_modal_key,
    check_environment_exists,
    create_environment,
)
from eve import deploy
from eve.tools.comfyui_tool import convert_tasks2_to_tasks3


logging.basicConfig(level=logging.INFO)
logging.getLogger("ably").setLevel(logging.WARNING)


db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
app_name = "api-prod" if db == "PROD" else "api-stage"


# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    watch_thread = threading.Thread(target=convert_tasks2_to_tasks3, daemon=True)
    watch_thread.start()
    app.state.watch_thread = watch_thread

    app.state.ably_client = AblyRealtime(
        os.getenv("ABLY_PUBLISHER_KEY"),
        options={
            "heartbeat_interval": 15000,
            "connection_state_ttl": 60000,
            "disconnected_retry_timeout": 15000,
        },
    )
    yield
    # Shutdown
    if hasattr(app.state, "watch_thread"):
        app.state.watch_thread.join(timeout=5)
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=True)
    if hasattr(app.state, "ably_client"):
        await app.state.ably_client.close()


web_app = FastAPI(lifespan=lifespan)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
scheduler = BackgroundScheduler()
scheduler.start()

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
background_tasks: BackgroundTasks = BackgroundTasks()


@web_app.post("/create")
async def create(request: TaskRequest, _: dict = Depends(auth.authenticate_admin)):
    return await handle_create(request)


@web_app.post("/cancel")
async def cancel(request: CancelRequest, _: dict = Depends(auth.authenticate_admin)):
    return await handle_cancel(request)


@web_app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_chat(request, background_tasks, web_app.state.ably_client)


@web_app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_stream_chat(request, background_tasks)


@web_app.post("/deployment")
async def deployment(
    request: DeployRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_deployment(request)


@web_app.post("/schedule")
async def schedule(
    request: ScheduleRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_schedule(request)


# Modal app setup
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

root_dir = Path(__file__).parent.parent
workflows_dir = root_dir / ".." / "workflows"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE")})
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .run_commands(["playwright install"])
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
    authenticate_modal_key()
    if not check_environment_exists(deploy.DEPLOYMENT_ENV_NAME):
        create_environment(deploy.DEPLOYMENT_ENV_NAME)
    return web_app
