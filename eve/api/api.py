from contextlib import asynccontextmanager
import os
import modal
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
import logging
from ably import AblyRealtime
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path

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

# Config and logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
app_name = "api-prod" if db == "PROD" else "api-stage"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    scheduler = BackgroundScheduler()
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.ably_client = AblyRealtime(os.getenv("ABLY_PUBLISHER_KEY"))
    yield
    # Cleanup
    if hasattr(app.state, "ably_client"):
        app.state.ably_client.close()
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown()


# FastAPI setup
web_app = FastAPI(lifespan=lifespan)
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
    request: ScheduleRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_schedule(
        request=request,
        background_tasks=background_tasks,
        scheduler=web_app.state.scheduler,
        ably_client=web_app.state.ably_client,
    )


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
