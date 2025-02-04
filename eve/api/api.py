import logging
import os
import threading
import json
from fastapi.responses import JSONResponse
import modal
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk

from eve import auth, db
from eve.postprocessing import (
    generate_lora_thumbnails,
    cancel_stuck_tasks,
    download_nsfw_models,
)
from eve.api.handlers import (
    handle_create,
    handle_cancel,
    handle_replicate_webhook,
    handle_chat,
    handle_deployment_create,
    handle_deployment_delete,
    handle_stream_chat,
    handle_trigger_create,
    handle_trigger_delete,
    handle_deployment_configure,
)
from eve.api.api_requests import (
    CancelRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    TaskRequest,
    ConfigureDeploymentRequest,
)
from eve.deploy import (
    authenticate_modal_key,
    check_environment_exists,
    create_environment,
)
from eve.tools.comfyui_tool import convert_tasks2_to_tasks3
from eve import deploy
from eve.tool import Tool
from eve.trigger import load_existing_triggers


app_name = f"api-{db.lower()}"
logging.basicConfig(level=logging.INFO)
logging.getLogger("ably").setLevel(logging.INFO if db != "PROD" else logging.WARNING)


# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    Tool.init_handler_cache()
    watch_thread = threading.Thread(target=convert_tasks2_to_tasks3, daemon=True)
    watch_thread.start()
    app.state.watch_thread = watch_thread

    # Initialize scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()
    app.state.scheduler = scheduler

    # Load existing triggers
    print("MODAL_SERVE", os.getenv("MODAL_SERVE"))
    should_load_triggers = (
        os.getenv("MODAL_SERVE") == "true"
        or os.getenv("LOAD_EXISTING_TRIGGERS") == "true"
    )
    if should_load_triggers:
        print("Loading existing triggers...")
        await load_existing_triggers(scheduler, handle_chat)

    try:
        yield
    finally:
        if hasattr(app.state, "watch_thread"):
            app.state.watch_thread.join(timeout=5)
        if hasattr(app.state, "scheduler"):
            app.state.scheduler.shutdown(wait=True)


class SentryContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("package", "eve-api")

            # Extract client context from headers
            client_platform = request.headers.get("X-Client-Platform")
            client_agent = request.headers.get("X-Client-Agent")

            if client_platform:
                scope.set_tag("client_platform", client_platform)
            if client_agent:
                scope.set_tag("client_agent", client_agent)

            scope.set_context(
                "api",
                {
                    "endpoint": request.url.path,
                    "modal_serve": os.getenv("MODAL_SERVE"),
                    "client_platform": client_platform,
                    "client_agent": client_agent,
                },
            )
        return await call_next(request)


web_app = FastAPI(lifespan=lifespan)
web_app.add_middleware(SentryContextMiddleware)
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


@web_app.post("/update")
async def replicate_webhook(request: Request):
    # Get raw body for signature verification
    body = await request.body()
    print(body)

    # Parse JSON body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON body"}

    # todo: validate webhook signature
    try:
        # headers = dict(request.headers)
        # secret = replicate.webhooks.default.secret()
        # replicate.webhooks.validate(
        #     body=body,  # Pass raw body for signature verification
        #     headers=headers,
        #     secret=secret
        # )
        pass
    except Exception as e:
        return {"status": "error", "message": f"Invalid webhook signature: {str(e)}"}

    return await handle_replicate_webhook(data)


@web_app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_chat(request, background_tasks)


@web_app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_stream_chat(request, background_tasks)


@web_app.post("/deployments/configure")
async def deployment_configure(
    request: ConfigureDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_deployment_configure(request)


@web_app.post("/deployments/create")
async def deployment_create(
    request: CreateDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_deployment_create(request)


@web_app.post("/deployments/delete")
async def deployment_delete(
    request: DeleteDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_deployment_delete(request)


@web_app.post("/triggers/create")
async def trigger_create(
    request: CreateTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_create(request, web_app.state.scheduler)


@web_app.post("/triggers/delete")
async def trigger_delete(
    request: DeleteTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_delete(request, web_app.state.scheduler)


@web_app.exception_handler(Exception)
async def catch_all_exception_handler(request, exc):
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )


# Modal app setup
app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

root_dir = Path(__file__).parent.parent.parent
workflows_dir = root_dir / ".." / "workflows"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE")})
    .apt_install(
        "git",
        "libmagic1",
        "ffmpeg",
        "wget",
        # Add Playwright dependencies
        "libnss3",
        "libnspr4",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libatspi2.0-0",
        "libxcomposite1",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("numpy<2.0", "torch==2.0.1", "torchvision", "transformers", "Pillow")
    .run_commands(["playwright install"])
    .run_function(download_nsfw_models)
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


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def postprocessing():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        print(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)

    # try:
    #     await run_nsfw_detection()
    # except Exception as e:
    #     print(f"Error running nsfw detection: {e}")
    #     sentry_sdk.capture_exception(e)

    try:
        await generate_lora_thumbnails()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)
