import logging
import os
import json
import modal
import replicate
import sentry_sdk
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pathlib import Path
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

from eve.api.handlers import (
    handle_create,
    handle_cancel,
    handle_discord_emission,
    handle_prompt_session,
    handle_replicate_webhook,
    handle_chat,
    handle_stream_chat,
    handle_telegram_emission,
    handle_telegram_update,
    handle_trigger_create,
    handle_trigger_delete,
    handle_trigger_stop,
    handle_twitter_update,
    handle_trigger_get,
    handle_agent_tools_update,
    handle_agent_tools_delete,
    handle_farcaster_update,
    handle_farcaster_emission,
    handle_session_cancel,
    handle_v2_deployment_create,
    handle_v2_deployment_emission,
    handle_v2_deployment_interact,
    handle_v2_deployment_update,
    handle_v2_deployment_delete,
    handle_v2_deployment_farcaster_neynar_webhook,
)
from eve.api.api_requests import (
    CancelRequest,
    CancelSessionRequest,
    ChatRequest,
    CreateDeploymentRequestV2,
    CreateTriggerRequest,
    DeleteDeploymentRequestV2,
    DeleteTriggerRequest,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    PromptSessionRequest,
    TaskRequest,
    PlatformUpdateRequest,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
    UpdateDeploymentRequestV2,
)
from eve.api.helpers import pre_modal_setup, busy_state_dict


app_name = f"api-{db.lower()}"
logging.getLogger("ably").setLevel(logging.INFO if db != "PROD" else logging.WARNING)

logger = logging.getLogger(__name__)


# FastAPI setup


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


web_app = FastAPI()
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


@web_app.post("/triggers/create")
async def trigger_create(
    request: CreateTriggerRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    pre_modal_setup()
    return await handle_trigger_create(request, background_tasks)


@web_app.post("/triggers/stop")
async def trigger_stop(
    request: DeleteTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_stop(request)


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
    request: Request,
):
    return await handle_farcaster_update(request)


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


@web_app.post("/emissions/platform/farcaster")
async def emissions_farcaster(
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_farcaster_emission(request)


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


@web_app.post("/sessions/cancel")
async def cancel_session(
    request: CancelSessionRequest,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_session_cancel(request)


@web_app.get("/sessions/{session_id}/stream")
async def stream_session(
    session_id: str,
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    from eve.api.handlers import handle_session_stream

    return await handle_session_stream(session_id, request)


@web_app.post("/v2/deployments/create")
async def create_deployment(
    request: CreateDeploymentRequestV2,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_v2_deployment_create(request)


@web_app.post("/v2/deployments/update")
async def update_deployment(
    request: UpdateDeploymentRequestV2,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_v2_deployment_update(request)


@web_app.post("/v2/deployments/delete")
async def delete_deployment(
    request: DeleteDeploymentRequestV2,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_v2_deployment_delete(request)


@web_app.get("/v2/deployments/interact")
async def deployment_interact(
    request: DeploymentInteractRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_v2_deployment_interact(request)


@web_app.post("/v2/deployments/farcaster/neynar-webhook")
async def deployment_farcaster_neynar_webhook(request: Request):
    return await handle_v2_deployment_farcaster_neynar_webhook(request)


@web_app.post("/v2/deployments/emission")
async def deployment_emission(request: DeploymentEmissionRequest):
    return await handle_v2_deployment_emission(request)


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
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    # .pip_install("numpy<2.0", "torch==2.0.1", "torchvision", "transformers", "Pillow")
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
    timeout=3600 * 3,  # 3 hours
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


@app.function(image=image, max_containers=10, timeout=3600)
@modal.concurrent(max_inputs=4)
async def run(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    result = await handler(args, user, agent)
    return eden_utils.upload_result(result)


@app.function(image=image, max_containers=10, timeout=3600)
@modal.concurrent(max_inputs=4)
@task_handler_func
async def run_task(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    return await handler(args, user, agent)


@app.function(image=image, max_containers=10, timeout=3600)
@modal.concurrent(max_inputs=4)
async def run_task_replicate(task: Task):
    task.update(status="running")
    tool = Tool.load(task.tool)
    n_samples = task.args.get("n_samples", 1)
    replicate_model = tool._get_replicate_model(task.args)
    args = tool.prepare_args(task.args)
    args = tool._format_args_for_replicate(args)
    try:
        outputs = []
        for i in range(n_samples):
            task_args = args.copy()
            if "seed" in task_args:
                task_args["seed"] = task_args["seed"] + i
            output = await replicate.async_run(replicate_model, input=task_args)
            outputs.append(output)
        outputs = flatten_list(outputs)
        result = replicate_update_task(task, "succeeded", None, outputs, "normal")
    except Exception as e:
        print(f"Error running replicate: {e}")
        sentry_sdk.capture_exception(e)
        result = replicate_update_task(task, "failed", str(e), None, "normal")
    return result


@app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=5), timeout=300
)
async def cleanup_stale_session_clients():
    """Clean up stale SSE clients from session streams"""
    try:
        from eve.api.session_streams import session_stream_manager

        logger.info("Starting stale session client cleanup...")
        await session_stream_manager.cleanup_stale_clients(
            max_age_seconds=3600
        )  # 1 hour
        logger.info("Finished cleaning up stale session clients")
    except Exception as e:
        logger.error(f"Error in cleanup_stale_session_clients: {e}")
        sentry_sdk.capture_exception(e)


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


def flatten_list(seq):
    """Flattens a list that is either flat or nested one level deep."""
    return [x for item in seq for x in (item if isinstance(item, list) else [item])]
