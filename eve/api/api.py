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

from eve import auth, db, eden_utils
from eve.api.runner_tasks import (
    cancel_stuck_tasks,
    generate_lora_thumbnails,
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
    handle_prompt_session,
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
    handle_agent_tools_update,
    handle_agent_tools_delete,
)
from eve.api.api_requests import (
    CancelRequest,
    ChatRequest,
    CreateDeploymentRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequest,
    DeleteTriggerRequest,
    PromptSessionRequest,
    TaskRequest,
    PlatformUpdateRequest,
    UpdateDeploymentRequest,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
)
from eve.api.helpers import pre_modal_setup, busy_state_dict


app_name = f"api-{db.lower()}"
logging.getLogger("ably").setLevel(logging.INFO if db != "PROD" else logging.WARNING)

logger = logging.getLogger(__name__)


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
    request: CreateTriggerRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    pre_modal_setup()
    return await handle_trigger_create(request, background_tasks)


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


@web_app.post("/agent/tools/update")
async def agent_tools_update(
    request: AgentToolsUpdateRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_agent_tools_update(request)


@web_app.post("/agent/tools/delete")
async def agent_tools_delete(
    request: AgentToolsDeleteRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_agent_tools_delete(request)


@web_app.post("/sessions/prompt")
async def prompt_session(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_prompt_session(request, background_tasks)


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
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE", "False")})
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
        "libgtk-3-0",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    # .pip_install("numpy<2.0", "torch==2.0.1", "torchvision", "transformers", "Pillow")
    .run_commands(["playwright install"])
    # .run_function(download_nsfw_models)
    .add_local_dir(str(workflows_dir), "/workflows")
    .add_local_file(str(root_dir / "pyproject.toml"), "/eve/pyproject.toml")
    .add_local_python_source("eve", ignore=[])
    .add_local_python_source("api", ignore=[])
)


@app.function(
    image=image,
    min_containers=1,
    max_containers=10,
    scaledown_window=60,
    timeout=3600,
)
@modal.concurrent(max_inputs=25)
@modal.asgi_app()
def fastapi_app():
    return web_app


@app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def cancel_stuck_tasks_fn():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        print(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)


# @app.function(
#     image=image,
#     max_containers=1,
#     schedule=modal.Period(minutes=15),
#     timeout=3600
# )
# async def run_nsfw_detection_fn():
#     try:
#         await run_nsfw_detection()
#     except Exception as e:
#         print(f"Error running nsfw detection: {e}")
#         sentry_sdk.capture_exception(e)


@app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def generate_lora_thumbnails_fn():
    try:
        await generate_lora_thumbnails()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, max_containers=1, schedule=modal.Period(hours=2), timeout=3600
)
async def rotate_agent_metadata_fn():
    try:
        await rotate_agent_metadata()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, max_containers=10, timeout=3600
)
@modal.concurrent(max_inputs=4)
async def run(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    result = await handler(args, user, agent)
    return eden_utils.upload_result(result)


@app.function(
    image=image, max_containers=10, timeout=3600
)
@modal.concurrent(max_inputs=4)
@task_handler_func
async def run_task(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    return await handler(args, user, agent)


@app.function(
    image=image, max_containers=10, timeout=3600
)
@modal.concurrent(max_inputs=4)
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
    max_containers=10,
    scaledown_window=60,
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
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
    image=image, max_containers=1, schedule=modal.Period(minutes=2), timeout=3600
)
async def cleanup_stale_busy_states():
    """Clean up any stale busy states in the shared modal.Dict"""
    try:
        current_time = time.time()
        stale_threshold = 300
        logger.info("Starting stale busy state cleanup...")

        # Get all keys from the dictionary first
        all_keys = list(busy_state_dict.keys())  # This is not atomic but necessary
        all_values = list(busy_state_dict.values())
        print(f"Checking keys: {all_keys}")
        print(f"Checking values: {all_values}")

        for key in all_keys:
            try:
                # Get current state
                current_state = busy_state_dict.get(key)
                # Check if state exists and is a dictionary with expected structure
                if (
                    not current_state
                    or not isinstance(current_state, dict)
                    or not all(
                        k in current_state
                        for k in ["requests", "timestamps", "context_map"]
                    )
                ):
                    logger.warning(
                        f"Removing invalid/stale state for key {key}: {current_state}"
                    )
                    # Delete directly if possible and safe
                    if key in busy_state_dict:
                        busy_state_dict.pop(key)
                    continue

                requests = current_state.get("requests", [])
                timestamps = current_state.get("timestamps", {})
                context_map = current_state.get("context_map", {})

                # Ensure correct types after retrieval
                requests = list(requests)
                timestamps = dict(timestamps)
                context_map = dict(context_map)

                stale_requests = []
                active_requests = []
                updated_timestamps = {}
                updated_context_map = {}

                # Iterate over a copy of request IDs
                for request_id in list(requests):
                    timestamp = timestamps.get(request_id, 0)
                    if current_time - timestamp > stale_threshold:
                        stale_requests.append(request_id)
                        logger.info(
                            f"Marking request {request_id} as stale for key {key} (age: {current_time - timestamp:.1f}s)."
                        )
                    else:
                        active_requests.append(request_id)
                        if request_id in timestamps:
                            updated_timestamps[request_id] = timestamps[request_id]
                        if request_id in context_map:
                            updated_context_map[request_id] = context_map[request_id]

                # If any requests were found to be stale, update the state
                if stale_requests:
                    logger.info(
                        f"Cleaning up {len(stale_requests)} stale requests for {key}. Original count: {len(requests)}"
                    )
                    # Update the state in the modal.Dict
                    if not active_requests:
                        # If no active requests left, remove the whole key
                        logger.info(
                            f"Removing key '{key}' as no active requests remain after cleanup."
                        )
                        if key in busy_state_dict:  # Check existence before deleting
                            busy_state_dict.pop(key)
                    else:
                        # Otherwise, update with cleaned lists/dicts
                        new_state = {
                            "requests": active_requests,
                            "timestamps": updated_timestamps,
                            "context_map": updated_context_map,
                        }
                        busy_state_dict.put(key, new_state)
                        logger.info(
                            f"Updated state for key '{key}'. Active requests: {len(active_requests)}"
                        )
                # else: # No stale requests found for this key
                #    logger.debug(f"No stale requests found for key '{key}'.")
            except KeyError:
                logger.warning(
                    f"Key {key} was deleted concurrently during cleanup processing."
                )
                continue  # Key was likely deleted by another process or previous step
            except Exception as key_e:
                logger.error(
                    f"Error processing key '{key}' during cleanup: {key_e}",
                    exc_info=True,
                )
                # Decide how to handle errors: skip key, mark for later deletion, etc.
                # For now, just log and continue to avoid breaking the whole job.

        logger.info("Finished cleaning up stale busy states.")
    except Exception as e:
        logger.error(f"Error in cleanup_stale_busy_states job: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
