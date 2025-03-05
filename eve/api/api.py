import logging
import os
import threading
import json
import modal
import replicate
import sentry_sdk
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pathlib import Path
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.exceptions import RequestValidationError
import time
import asyncio

from eve import auth, db, eden_utils
from eve.api.runner_tasks import (
    cancel_stuck_tasks,
    download_nsfw_models,
    generate_lora_thumbnails,
    run_nsfw_detection,
    rotate_agent_metadata,
)
from eve.task import task_handler_func, Task
from eve.tool import Tool
from eve.tools.tool_handlers import load_handler
from eve.tools.replicate_tool import replicate_update_task
from eve.tools.comfyui_tool import convert_tasks2_to_tasks3

from eve.api.handlers import (
    handle_create,
    handle_cancel,
    handle_deployment_update,
    handle_discord_emission,
    handle_replicate_webhook,
    handle_chat,
    handle_deployment_create,
    handle_deployment_delete,
    handle_stream_chat,
    handle_telegram_emission,
    handle_telegram_update,
    handle_trigger_create,
    handle_trigger_delete,
    handle_twitter_update,
    handle_trigger_get,
)
from eve.api.api_requests import (
    CancelRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    PlatformUpdateRequest,
    TaskRequest,
    UpdateDeploymentRequest,
)
from eve.api.helpers import pre_modal_setup, busy_state


app_name = f"api-{db.lower()}"
logging.basicConfig(level=logging.INFO)
logging.getLogger("ably").setLevel(logging.INFO if db != "PROD" else logging.WARNING)


def load_watch_thread():
    watch_thread = threading.Thread(target=convert_tasks2_to_tasks3, daemon=True)
    watch_thread.start()
    return watch_thread


# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.watch_thread = load_watch_thread()
    try:
        yield
    finally:
        if hasattr(app.state, "watch_thread"):
            app.state.watch_thread.join(timeout=5)


class SentryContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("package", "eve-api")

            # Extract client context from headers
            client_platform = request.headers.get("X-Client-Platform")
            client_deployment_id = request.headers.get("X-Client-Deployment-Id")
            if client_platform:
                scope.set_tag("client_platform", client_platform)
            if client_deployment_id:
                scope.set_tag("client_deployment_id", client_deployment_id)

            scope.set_context(
                "api",
                {
                    "endpoint": request.url.path,
                    "modal_serve": os.getenv("MODAL_SERVE"),
                    "client_platform": client_platform,
                    "client_deployment_id": client_deployment_id,
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
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON body"}

    # Validate webhook signature
    try:
        body = body.decode()
        headers = dict(request.headers)
        secret = replicate.webhooks.default.secret()
        replicate.webhooks.validate(body=body, headers=headers, secret=secret)

    except Exception as e:
        print(f"Webhook validation failed: {str(e)}")
        return {"status": "error", "message": f"Invalid webhook signature: {str(e)}"}

    return await handle_replicate_webhook(data)


@web_app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    print("CHAT", request)
    return await handle_chat(request, background_tasks)


@web_app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_stream_chat(request, background_tasks)


@web_app.post("/deployments/create")
async def deployment_create(
    request: CreateDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_deployment_create(request)


@web_app.post("/deployments/update")
async def deployment_update(
    request: UpdateDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_deployment_update(request)


@web_app.post("/deployments/delete")
async def deployment_delete(
    request: DeleteDeploymentRequest, _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_deployment_delete(request)


@web_app.post("/triggers/create")
async def trigger_create(
    request: CreateTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_create(request)


@web_app.post("/triggers/delete")
async def trigger_delete(
    request: DeleteTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_delete(request)


@web_app.post("/updates/platform/telegram")
async def updates_telegram(
    request: Request,
):
    return await handle_telegram_update(request)


@web_app.post("/updates/platform/farcaster")
async def updates_farcaster(
    request: PlatformUpdateRequest,
    _: dict = Depends(auth.authenticate_admin),
):
    return {"status": "success"}


@web_app.post("/updates/platform/twitter")
async def updates_twitter(
    request: PlatformUpdateRequest,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_twitter_update(request)


@web_app.post("/emissions/platform/discord")
async def emissions_discord(
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_discord_emission(request)


@web_app.post("/emissions/platform/telegram")
async def emissions_telegram(
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_telegram_emission(request)


@web_app.get("/triggers/{trigger_id}")
async def trigger_get(trigger_id: str, _: dict = Depends(auth.authenticate_admin)):
    return await handle_trigger_get(trigger_id)


@web_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error on {request.url}:")
    print(f"Request body: {await request.body()}")
    print(f"Validation errors: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": await request.json() if await request.body() else None,
        },
    )


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
    .add_local_dir(str(workflows_dir), "/workflows")
    .add_local_file(str(root_dir / "pyproject.toml"), "/eve/pyproject.toml")
)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=10,
    container_idle_timeout=60,
    allow_concurrent_inputs=25,
    timeout=3600,
)
@modal.asgi_app()
def fastapi_app():
    return web_app


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def cancel_stuck_tasks_fn():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        print(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def run_nsfw_detection_fn():
    try:
        await run_nsfw_detection()
    except Exception as e:
        print(f"Error running nsfw detection: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def generate_lora_thumbnails_fn():
    try:
        await generate_lora_thumbnails()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(hours=2), timeout=3600
)
async def rotate_agent_metadata_fn():
    try:
        await rotate_agent_metadata()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=10, allow_concurrent_inputs=4, timeout=3600
)
async def run(tool_key: str, args: dict):
    handler = load_handler(tool_key)
    result = await handler(args)
    return eden_utils.upload_result(result)


@app.function(
    image=image, concurrency_limit=10, allow_concurrent_inputs=4, timeout=3600
)
@task_handler_func
async def run_task(tool_key: str, args: dict, user: str = None, requester: str = None):
    handler = load_handler(tool_key)
    return await handler(args, user, requester)


@app.function(
    image=image, concurrency_limit=10, allow_concurrent_inputs=4, timeout=3600
)
async def run_task_replicate(task: Task):
    task.update(status="running")
    tool = Tool.load(task.tool)
    args = tool.prepare_args(task.args)
    args = tool._format_args_for_replicate(args)
    replicate_model = tool._get_replicate_model(task.args)
    output = await replicate.async_run(replicate_model, input=args)
    result = replicate_update_task(task, "succeeded", None, output, "normal")
    return result


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=10,
    container_idle_timeout=60,
    allow_concurrent_inputs=10,
    timeout=3600,
)
async def deploy_client_modal(
    agent_id: str,
    agent_key: str,
    platform: str,
    secrets: dict,
    env: str,
    repo_branch: str = None,
):
    from eve.deploy import deploy_client as deploy_client_impl, DeploymentSecrets

    secrets_model = DeploymentSecrets(**secrets)

    return deploy_client_impl(
        agent_id=agent_id,
        agent_key=agent_key,
        platform=platform,
        secrets=secrets_model,
        env=env,
        repo_branch=repo_branch,
    )


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=2), timeout=3600
)
async def cleanup_stale_busy_states():
    """Clean up any stale busy states that might be lingering"""
    try:
        current_time = time.time()
        stale_threshold = 240

        # Get all keys in the busy_state dict
        all_keys = list(busy_state.keys())

        for key in all_keys:
            # Get the timestamps for this key
            timestamps = busy_state.get(f"{key}_timestamps", {})

            # Get the request IDs for this key
            requests = busy_state.get(key, [])

            # Check each request ID
            stale_requests = []
            for request_id in requests:
                timestamp = timestamps.get(request_id, 0)
                if current_time - timestamp > stale_threshold:
                    stale_requests.append(request_id)

            # Remove stale requests
            if stale_requests:
                updated_requests = [r for r in requests if r not in stale_requests]
                busy_state[key] = updated_requests

                # Also update timestamps
                updated_timestamps = {
                    k: v for k, v in timestamps.items() if k not in stale_requests
                }
                busy_state[f"{key}_timestamps"] = updated_timestamps

                print(f"Cleaned up {len(stale_requests)} stale requests for {key}")

                # If this was a Discord or Telegram deployment, emit typing stop
                if "discord" in key or "telegram" in key:
                    deployment_id, platform = key.split(".")

                    if platform == "discord":
                        # Find any channel IDs associated with this deployment
                        from eve.api.helpers import emit_typing_update

                        for channel_id in busy_state.get(f"{key}_channels", []):
                            asyncio.create_task(
                                emit_typing_update(deployment_id, channel_id, False)
                            )

                    elif platform == "telegram":
                        # Find any chat IDs associated with this deployment
                        from eve.api.helpers import emit_telegram_typing_update

                        for chat_key in busy_state.get(f"{key}_chats", []):
                            chat_id, thread_id = (
                                chat_key.split("_")
                                if "_" in chat_key
                                else (chat_key, None)
                            )
                            asyncio.create_task(
                                emit_telegram_typing_update(
                                    deployment_id, chat_id, thread_id, False
                                )
                            )

        print("Finished cleaning up stale busy states")
    except Exception as e:
        print(f"Error cleaning up stale busy states: {e}")
        sentry_sdk.capture_exception(e)
