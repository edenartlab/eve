import asyncio
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import modal
import replicate
import sentry_sdk
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from eve import auth, db
from eve.agent.session.models import Session
from eve.agent.session.run import remote_prompt_session, run_automatic_session
from eve.api.api_functions import (
    cancel_stuck_tasks_fn,
    cleanup_stale_busy_states,
    embed_recent_creations,
    generate_lora_thumbnails_fn,
    rotate_agent_metadata_fn,
    run,
    run_task,
    run_task_replicate,
)
from eve.api.api_requests import (
    AgentPromptsExtractionRequest,
    AgentToolsDeleteRequest,
    AgentToolsUpdateRequest,
    CancelRequest,
    CancelSessionRequest,
    CreateConceptRequest,
    CreateDeploymentRequestV2,
    CreateNotificationRequest,
    CreateTriggerRequest,
    DeleteDeploymentRequestV2,
    DeleteTriggerRequest,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    EmbedSearchRequest,
    PromptSessionRequest,
    RegenerateAgentMemoryRequest,
    RegenerateUserMemoryRequest,
    RunTriggerRequest,
    TaskRequest,
    UpdateConceptRequest,
    UpdateDeploymentRequestV2,
    UpdateSessionStatusRequest,
)
from eve.api.handlers import (
    handle_agent_tools_delete,
    handle_agent_tools_update,
    handle_cancel,
    handle_create,
    handle_create_notification,
    handle_embedsearch,
    handle_extract_agent_prompts,
    handle_prompt_session,
    handle_regenerate_agent_memory,
    handle_regenerate_user_memory,
    handle_replicate_webhook,
    handle_session_cancel,
    handle_session_message,
    handle_session_run,
    handle_session_status_update,
    handle_v2_deployment_create,
    handle_v2_deployment_delete,
    handle_v2_deployment_email_inbound,
    handle_v2_deployment_emission,
    handle_v2_deployment_farcaster_neynar_webhook,
    handle_v2_deployment_interact,
    handle_v2_deployment_update,
)
from eve.api.runner_tasks import download_clip_models
from eve.concepts import (
    create_concept_thumbnail,
    handle_concept_create,
    handle_concept_update,
)
from eve.trigger import (
    Trigger,
    execute_trigger,
    handle_trigger_create,
    handle_trigger_delete,
    handle_trigger_get,
    handle_trigger_run,
    handle_trigger_stop,
)

app_name = f"api-{db.lower()}"
logging.getLogger("ably").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
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


# Global flag for shutdown
_shutdown_event = asyncio.Event()


def handle_shutdown_signal(signum, frame):
    """Handle SIGINT/SIGTERM to close SSE connections immediately"""
    from eve.api.sse_manager import sse_manager

    # Run async close in the event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(sse_manager.close_all())
    _shutdown_event.set()
    # Force exit after a short delay
    import os

    os._exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown - close all SSE connections
    from eve.api.sse_manager import sse_manager

    await sse_manager.close_all()


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
        return {"status": "error", "message": f"Invalid webhook signature: {str(e)}"}

    return await handle_replicate_webhook(data)


@web_app.post("/triggers/create")
async def trigger_create(
    request: CreateTriggerRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_trigger_create(request, background_tasks)


@web_app.post("/triggers/stop")
async def trigger_stop(
    request: DeleteTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_stop(request)


@web_app.post("/triggers/delete")
async def trigger_delete(
    request: DeleteTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_delete(request)


@web_app.post("/triggers/run")
async def trigger_run(
    request: RunTriggerRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_run(request)


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


@web_app.post("/concepts/create")
async def concept_create(
    request: CreateConceptRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_concept_create(request, background_tasks)


@web_app.post("/concepts/update")
async def concept_update(
    request: UpdateConceptRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_concept_update(request, background_tasks)


@web_app.post("/sessions/prompt")
async def prompt_session(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    """Add a message to a session and run orchestration (combined operation)."""
    return await handle_prompt_session(request, background_tasks)


@web_app.post("/sessions/message")
async def session_message(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    """Add a message to a session without running orchestration."""
    return await handle_session_message(request, background_tasks)


@web_app.post("/sessions/run")
async def session_run(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    """Run orchestration on a session without adding a message."""
    return await handle_session_run(request, background_tasks)


@web_app.post("/sessions/cancel")
async def cancel_session(
    request: CancelSessionRequest,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_session_cancel(request)


@web_app.post("/sessions/status")
async def update_session_status(
    request: UpdateSessionStatusRequest,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_session_status_update(request)


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


@web_app.post("/v2/deployments/email/inbound")
async def deployment_email_inbound(request: Request):
    return await handle_v2_deployment_email_inbound(request)


@web_app.post("/v2/deployments/emission")
async def deployment_emission(request: DeploymentEmissionRequest):
    return await handle_v2_deployment_emission(request)


# Notification routes
@web_app.post("/notifications/create")
async def create_notification(
    request: CreateNotificationRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_create_notification(request)


# Embed search route
@web_app.post("/embedsearch")
async def embedsearch(
    request: EmbedSearchRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_embedsearch(request)


# Agent creation - extract prompts from conversation session
@web_app.post("/agent_creation/extract_prompts")
async def extract_agent_prompts(
    request: AgentPromptsExtractionRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_extract_agent_prompts(request)


# Memory regeneration endpoints
@web_app.post("/memory/regenerate-agent-memory")
async def regenerate_agent_memory(
    request: RegenerateAgentMemoryRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_regenerate_agent_memory(request)


@web_app.post("/memory/regenerate-user-memory")
async def regenerate_user_memory(
    request: RegenerateUserMemoryRequest, _: dict = Depends(auth.authenticate_admin)
):
    return await handle_regenerate_user_memory(request)


# Development endpoints for local testing
@web_app.post("/dev/twitter/poll")
async def dev_poll_twitter(request: Request):
    """
    Local development endpoint for Twitter polling.
    Processes tweets synchronously instead of spawning Modal tasks.

    Usage:
        POST http://localhost:8000/dev/twitter/poll

    Optional query params:
        ?key=<secret> - Simple auth for dev endpoint
    """
    # Simple dev auth (optional)
    secret_key = request.query_params.get("key")
    expected_key = os.getenv("DEV_API_KEY", "dev")

    if secret_key != expected_key:
        return JSONResponse(
            status_code=401, content={"error": "Invalid or missing dev API key"}
        )

    from eve.agent.deployments.twitter import poll_twitter_gateway

    logger.info("=== Running Twitter poll in LOCAL MODE ===")
    result = await poll_twitter_gateway(local_mode=True)

    return JSONResponse(content=result)


@web_app.post("/dev/twitter/process_tweet")
async def dev_process_tweet(request: Request):
    """
    Local development endpoint to manually process a single tweet.

    Usage:
        POST http://localhost:8000/dev/twitter/process_tweet?key=dev

    Body (JSON):
    {
        "tweet_id": "1234567890",
        "deployment_id": "507f1f77bcf86cd799439011",
        "tweet_data": {
            "data": {...},
            "includes": {...}
        }
    }
    """
    # Simple dev auth
    secret_key = request.query_params.get("key")
    expected_key = os.getenv("DEV_API_KEY", "dev")

    if secret_key != expected_key:
        return JSONResponse(
            status_code=401, content={"error": "Invalid or missing dev API key"}
        )

    from eve.agent.deployments.twitter import process_twitter_tweet

    body = await request.json()
    tweet_id = body.get("tweet_id")
    deployment_id = body.get("deployment_id")
    tweet_data = body.get("tweet_data")

    if not all([tweet_id, deployment_id, tweet_data]):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Missing required fields: tweet_id, deployment_id, tweet_data"
            },
        )

    logger.info(f"=== Processing single tweet {tweet_id} in LOCAL MODE ===")
    result = await process_twitter_tweet(tweet_id, tweet_data, deployment_id)

    return JSONResponse(content=result)


# Simple embed endpoint that just returns the embedding vector
@web_app.post("/embed")
async def embed(request: Request, _: dict = Depends(auth.authenticate_admin)):
    """Generate CLIP embedding for a text query"""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    body = await request.json()
    query = body.get("query")
    if not query:
        return JSONResponse(
            status_code=400, content={"error": "query parameter is required"}
        )

    MODEL_NAME = "openai/clip-vit-large-patch14"
    device = "cpu"

    try:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        proc = CLIPProcessor.from_pretrained(MODEL_NAME)

        inputs = proc(
            text=[query], return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            v = model.get_text_features(**inputs)
            qv = F.normalize(v, p=2, dim=-1)[0].cpu().tolist()

        return {"embedding": qv}
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate embedding: {str(e)}"},
        )


@web_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url}:")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")

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
media_cache_vol = modal.Volume.from_name("media-cache", create_if_missing=True)

app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
        modal.Secret.from_name("abraham-secrets"),
    ],
)

root_dir = Path(__file__).parent.parent.parent
workflows_dir = root_dir / ".." / "workflows"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": "1"})
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .run_function(download_clip_models)
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
    volumes={"/data/media-cache": media_cache_vol},
)
@modal.concurrent(max_inputs=25)
@modal.asgi_app()
def fastapi_app():
    return web_app


cancel_stuck_tasks_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)(cancel_stuck_tasks_fn)

generate_lora_thumbnails_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=15), timeout=3600
)(generate_lora_thumbnails_fn)


rotate_agent_metadata_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(hours=2), timeout=3600
)(rotate_agent_metadata_fn)


# run_scheduled_triggers_modal = app.function(
#     image=image, max_containers=1, schedule=modal.Period(minutes=1), timeout=300
# )(run_scheduled_triggers_fn)

run = app.function(
    image=image,
    max_containers=10,
    timeout=3600,
    volumes={"/data/media-cache": media_cache_vol},
)(modal.concurrent(max_inputs=4)(run))


run_task = app.function(
    image=image,
    min_containers=2,
    max_containers=10,
    timeout=3600,
    volumes={"/data/media-cache": media_cache_vol},
)(modal.concurrent(max_inputs=4)(run_task))


run_task_replicate = app.function(
    image=image,
    min_containers=2,
    max_containers=10,
    timeout=3600,
    volumes={"/data/media-cache": media_cache_vol},
)(modal.concurrent(max_inputs=4)(run_task_replicate))


cleanup_stale_busy_states_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=2), timeout=3600
)(cleanup_stale_busy_states)


embed_recent_creations_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=5), timeout=600
)(embed_recent_creations)


########################################################
## Concepts
########################################################


create_concept_thumbnail = app.function(
    image=image,
    min_containers=1,
    max_containers=10,
    timeout=600,
    volumes={"/data/media-cache": media_cache_vol},
)(modal.concurrent(max_inputs=4)(create_concept_thumbnail))


########################################################
## Triggers
########################################################


@app.function(image=image, max_containers=4)
async def execute_trigger_fn(trigger_id: str) -> Session:
    return await execute_trigger(trigger_id)


@app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=2), timeout=300
)
async def run_scheduled_triggers_fn():
    current_time = datetime.now(timezone.utc)

    # Find active triggers where next_scheduled_run <= current time
    triggers = list(
        Trigger.find(
            {
                # "_id": ObjectId("68d634bc050ad90f48a03e6f"),
                "status": "active",
                "deleted": {"$ne": True},
                "next_scheduled_run": {"$lte": current_time},
            }
        )
    )

    logger.info(f"Found {len(triggers)} triggers to run")

    if not triggers:
        logger.info("No triggers to run")
        return

    sessions = []
    triggers = [str(trigger.id) for trigger in triggers]

    async for result in execute_trigger_fn.map.aio(triggers):
        sessions.append(result)

    logger.info(f"Ran {len(triggers)} triggers")
    logger.info(sessions)


########################################################
## Remote Session Prompting
########################################################


@app.function(image=image, max_containers=10, timeout=3600)
async def remote_prompt_session_fn(
    session_id: str,
    agent_id: str,
    user_id: str,
    content: str,
    attachments: Optional[List[str]] = [],
    extra_tools: Optional[List[str]] = [],
):
    return await remote_prompt_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=user_id,
        content=content,
        attachments=attachments,
        extra_tools=extra_tools,
    )


@app.function(image=image, max_containers=4, timeout=3600)
async def handle_session_status_change_fn(session_id: str, status: str):
    if status == "active":
        await run_automatic_session(session_id)


########################################################
## Farcaster Webhook Processing
########################################################


@app.function(image=image, max_containers=10, timeout=3600)
async def process_farcaster_cast_fn(
    cast_hash: str,
    cast_data: dict,
    deployment_id: str,
):
    """Modal wrapper for Farcaster cast processing"""
    from eve.agent.deployments.farcaster import process_farcaster_cast

    return await process_farcaster_cast(cast_hash, cast_data, deployment_id)


########################################################
## Twitter Polling Gateway
########################################################


@app.function(image=image, max_containers=10, timeout=3600)
async def process_twitter_tweet_fn(
    tweet_id: str,
    tweet_data: dict,
    deployment_id: str,
):
    """Modal wrapper for Twitter tweet processing"""
    from eve.agent.deployments.twitter import process_twitter_tweet

    return await process_twitter_tweet(tweet_id, tweet_data, deployment_id)


@app.function(
    image=image,
    max_containers=1,
    schedule=modal.Period(minutes=1),
    timeout=600,
)
async def poll_twitter_gateway_fn():
    """Modal scheduled function for Twitter polling gateway"""
    if os.getenv("DB") == "PROD":
        logger.info("Skipping Twitter polling in PROD")
        return

    from eve.agent.deployments.twitter import poll_twitter_gateway

    return await poll_twitter_gateway()
