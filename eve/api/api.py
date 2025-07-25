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

from eve import auth, db

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
    handle_create_notification,
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
    CreateNotificationRequest,
)
from eve.api.helpers import pre_modal_setup
from eve.api.api_functions import (
    cancel_stuck_tasks_fn,
    generate_lora_thumbnails_fn,
    rotate_agent_metadata_fn,
    run_scheduled_triggers_fn,
    run,
    run_task,
    run_task_replicate,
    cleanup_stale_busy_states,
)


app_name = f"api-{db.lower()}"
logging.getLogger("ably").setLevel(logging.WARNING)

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


# Notification routes
@web_app.post("/notifications/create")
async def create_notification(
    request: CreateNotificationRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_create_notification(request)


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


cancel_stuck_tasks_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)(cancel_stuck_tasks_fn)


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


generate_lora_thumbnails_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)(generate_lora_thumbnails_fn)


rotate_agent_metadata_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(hours=2), timeout=3600
)(rotate_agent_metadata_fn)


run_scheduled_triggers_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=1), timeout=300
)(run_scheduled_triggers_fn)


run = app.function(image=image, max_containers=10, timeout=3600)(
    modal.concurrent(max_inputs=4)(run)
)


run_task = app.function(image=image, max_containers=10, timeout=3600)(
    modal.concurrent(max_inputs=4)(run_task)
)


run_task_replicate = app.function(image=image, max_containers=10, timeout=3600)(
    modal.concurrent(max_inputs=4)(run_task_replicate)
)


cleanup_stale_busy_states_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=2), timeout=3600
)(cleanup_stale_busy_states)
