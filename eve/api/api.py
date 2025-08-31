import logging
import asyncio
import uuid
import os
import json
import signal
import threading
from typing import Dict, Set
import modal
import replicate
import sentry_sdk
from bson import ObjectId
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.exceptions import RequestValidationError
from datetime import datetime, timezone

from eve import auth, db
from eve.agent.session.triggers import calculate_next_scheduled_run
from eve.agent.agent import Agent
from eve.user import User
from eve.api.helpers import pre_modal_setup
from eve.api.handlers import (
    setup_session,
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
    handle_trigger_run,
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
    RunTriggerRequest,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    PromptSessionRequest,
    TaskRequest,
    PlatformUpdateRequest,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
    UpdateDeploymentRequestV2,
    CreateNotificationRequest,
    SessionCreationArgs
)
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
from eve.agent.session.models import (
    ChatMessageRequestInput, 
    LLMConfig,  
    PromptSessionContext, 
    Session,
    Trigger
)
from eve.agent.session.session import (
    add_user_message, 
    async_prompt_session, 
    build_llm_context
)




app_name = f"api-{db.lower()}"
logging.getLogger("ably").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global trigger execution tracking
running_triggers: Dict[str, Dict] = {}
trigger_lock = threading.Lock()


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
async def create(
    request: TaskRequest, 
    _: dict = Depends(auth.authenticate_admin)
):
    return await handle_create(request)
    


@web_app.post("/cancel")
async def cancel(
    request: CancelRequest, 
    _: dict = Depends(auth.authenticate_admin)
):
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
    request: DeleteTriggerRequest, 
    _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_stop(request)


@web_app.post("/triggers/delete")
async def trigger_delete(
    request: DeleteTriggerRequest, 
    _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_delete(request)


@web_app.post("/triggers/run")
async def trigger_run(
    request: RunTriggerRequest, 
    _: dict = Depends(auth.authenticate_admin)
):
    pre_modal_setup()
    return await handle_trigger_run(request)


@web_app.post("/triggers/interrupt/{trigger_id}")
async def trigger_interrupt(
    trigger_id: str,
    _: dict = Depends(auth.authenticate_admin)
):
    """Interrupt a running trigger execution"""
    try:
        with trigger_lock:
            if trigger_id not in running_triggers:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Trigger {trigger_id} is not currently running"}
                )
            
            # Request interrupt (will be picked up by signal handlers)
            interrupt_handlers[trigger_id] = True
            running_triggers[trigger_id]["status"] = "interrupting"
            
            logger.info(f"Interrupt requested for trigger {trigger_id}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Interrupt requested for trigger {trigger_id}",
                    "trigger_id": trigger_id,
                    "status": "interrupting"
                }
            )
                
    except Exception as e:
        logger.error(f"Error interrupting trigger {trigger_id}: {str(e)}")
        sentry_sdk.capture_exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to interrupt trigger: {str(e)}"}
        )


@web_app.get("/triggers/running")
async def get_running_triggers(_: dict = Depends(auth.authenticate_admin)):
    """Get list of currently running triggers"""
    try:
        with trigger_lock:
            running_list = []
            for trigger_id, info in running_triggers.items():
                running_list.append({
                    "trigger_id": trigger_id,
                    "trigger_name": info["trigger"].name,
                    "start_time": info["start_time"].isoformat(),
                    "status": info["status"]
                })
            
            return JSONResponse(
                status_code=200,
                content={
                    "running_triggers": running_list,
                    "count": len(running_list)
                }
            )
            
    except Exception as e:
        logger.error(f"Error getting running triggers: {str(e)}")
        sentry_sdk.capture_exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get running triggers: {str(e)}"}
        )


@web_app.post("/updates/platform/telegram")
async def updates_telegram(
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_telegram_update(request)


@web_app.post("/updates/platform/farcaster")
async def updates_farcaster(
    request: Request,
    _: dict = Depends(auth.authenticate_admin),
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
async def trigger_get(
    trigger_id: str, 
    _: dict = Depends(auth.authenticate_admin)
):
    return await handle_trigger_get(trigger_id)


@web_app.post("/agent/tools/update")
async def agent_tools_update(
    request: AgentToolsUpdateRequest,
    _: dict = Depends(auth.authenticate_admin)
):
    return await handle_agent_tools_update(request)


@web_app.post("/agent/tools/delete")
async def agent_tools_delete(
    request: AgentToolsDeleteRequest, 
    _: dict = Depends(auth.authenticate_admin)
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
    request: DeploymentInteractRequest, 
    _: dict = Depends(auth.authenticate_admin)
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
    request: CreateNotificationRequest, 
    _: dict = Depends(auth.authenticate_admin)
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
media_cache_vol = modal.Volume.from_name("media-cache", create_if_missing=True)

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
    volumes={"/data/media-cache": media_cache_vol},
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


run = app.function(
    image=image, max_containers=10, timeout=3600, volumes={"/data/media-cache": media_cache_vol}
)(modal.concurrent(max_inputs=4)(run))


run_task = app.function(
    image=image, min_containers=2, max_containers=10, timeout=3600, volumes={"/data/media-cache": media_cache_vol}
)(modal.concurrent(max_inputs=4)(run_task))


run_task_replicate = app.function(
    image=image, min_containers=2, max_containers=10, timeout=3600, volumes={"/data/media-cache": media_cache_vol}
)(modal.concurrent(max_inputs=4)(run_task_replicate)
)


cleanup_stale_busy_states_modal = app.function(
    image=image, max_containers=1, schedule=modal.Period(minutes=2), timeout=3600
)(cleanup_stale_busy_states)






# # local entrypoint
# @web_app.post("/triggers/letsgo")
# async def run_all_tasks(
#     request: CreateTriggerRequest,
#     background_tasks: BackgroundTasks,
#     _: dict = Depends(auth.authenticate_admin),
# ):
#     pre_modal_setup()
#     print("this is done")
#     # return await handle_trigger_create(request, background_tasks)




























async def handle_trigger_posting(trigger, session_id):
    """Handle posting instructions for a trigger"""
    import aiohttp

    posting_instructions = trigger.posting_instructions
    if not posting_instructions:
        return

    try:
        request_data = {
            "session_id": session_id,
            "user_id": str(trigger.user),
            "actor_agent_ids": [str(trigger.agent)],
            "message": {
                "role": "system",
                "content": f"""## Posting instructions
{posting_instructions.get("post_to", "")} channel {posting_instructions.get("channel_id", "")}

{posting_instructions.get("custom_instructions", "")}
""",
            },
            "update_config": trigger.update_config,
        }

        # Add custom tools based on platform
        platform = posting_instructions.get("post_to")
        if platform == "discord" and posting_instructions.get("channel_id"):
            request_data["custom_tools"] = {
                "discord_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }
        elif platform == "telegram" and posting_instructions.get("channel_id"):
            request_data["custom_tools"] = {
                "telegram_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }

        # Make async HTTP POST to prompt session endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to run posting instructions for trigger {trigger.trigger_id}: {error_text}"
                    )

    except Exception as e:
        logger.error(
            f"Error handling posting instructions for trigger {trigger.trigger_id}: {str(e)}"
        )
        sentry_sdk.capture_exception(e)






    


# Global interrupt handler for API endpoints
interrupt_handlers: Dict[str, bool] = {}


# Interruptible wrapper for trigger execution
@app.function(image=image, max_containers=4)
async def execute_trigger(trigger: Trigger) -> Session:
    """Interruptible trigger execution using existing triggers.py logic"""
    import signal
    from eve.agent.session.triggers import execute_trigger as base_execute_trigger
    from eve.agent.session.session import add_user_message
    
    trigger_id = trigger.trigger_id
    shutdown_requested = False
    session = None
    
    def signal_handler(signum, _):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info(f"Shutdown signal {signum} received for trigger {trigger_id}")
    
    def check_interrupt():
        nonlocal shutdown_requested
        if trigger_id in interrupt_handlers and interrupt_handlers[trigger_id]:
            shutdown_requested = True
            logger.info(f"API interrupt received for trigger {trigger_id}")
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Track this execution
    with trigger_lock:
        running_triggers[trigger_id] = {
            "trigger": trigger,
            "start_time": datetime.now(timezone.utc),
            "status": "running"
        }
    
    try:
        logger.info(f"Executing trigger {trigger_id} with name: '{trigger.name}'")
        
        # Check for interrupts before starting
        check_interrupt()
        if shutdown_requested:
            logger.info(f"Shutdown requested before execution for trigger {trigger_id}")
            return None
        
        # Use existing trigger execution logic
        result = await base_execute_trigger(trigger, is_immediate=False)
        session_id = result.get("session_id") if result else None
        
        # Get session object if execution completed
        if session_id:
            from eve.agent.session.models import Session as SessionModel
            session = SessionModel.from_mongo(ObjectId(session_id))

        return session

    except Exception as e:
        logger.error(f"Error executing trigger {trigger_id}: {str(e)}")
        sentry_sdk.capture_exception(e)
        return session
    
    finally:
        # Always perform cleanup
        try:
            if shutdown_requested and session:
                # Add interruption message to session
                from eve.agent.session.models import ChatMessageRequestInput, PromptSessionContext
                message = ChatMessageRequestInput(
                    role="user",
                    content="The user has interrupted this task."
                )
                context = PromptSessionContext(
                    session=session,
                    initiating_user_id=str(session.owner),
                    message=message,
                )
                add_user_message(session, context)
                session.save()
            
            # Remove from tracking
            with trigger_lock:
                running_triggers.pop(trigger_id, None)
                interrupt_handlers.pop(trigger_id, None)  # Clean up interrupt flag
            
            logger.info(f"Trigger execution cleanup completed for {trigger_id}")
            
        except Exception as e:
            logger.error(f"Error during final cleanup for trigger {trigger_id}: {e}")
    

# from eve.agent.session.models import Trigger
# async def execute_trigger_wrapper(trigger):
#     try:
#         from eve.agent.session.triggers import execute_trigger
        
#         # Use the shared execution function
#         result = await execute_trigger(trigger, is_immediate=False)
#         session_id = result.get("session_id")

#         # Update trigger with session if it was created


async def run_scheduled_triggers_fn2():
    """Check for and run scheduled triggers every minute"""

    current_time = datetime.now(timezone.utc)

    # Find active triggers which should be run now
    # triggers = Trigger.find({
    #     "status": "active",
    #     "deleted": {"$ne": True},
    #     # "next_scheduled_run": {"$lte": current_time},
    # })


    triggers = [Trigger.from_mongo(ObjectId(t)) for t in ["68b3bca333da060a73cef02a", "68b3bc8f33da060a73cef029"]]    

    triggers = [Trigger(**t.model_dump()) for t in triggers]

    print("THE NUMBER OF TRIGGERS TO RUN", len(triggers))

    async for result in execute_trigger.map.aio(triggers):
        print("trigger result", result)




# run_all_tasks_modal = app.function(
#     image=image, max_containers=1, #schedule=modal.Period(minutes=2), timeout=3600
# )(run_scheduled_triggers_fn2)


@app.local_entrypoint()
async def this_is_a_test():
    print("ok3 run a task??..")
    await run_scheduled_triggers_fn2()
    print("done!")


# if __name__ == "__main__":
#     asyncio.run(run_scheduled_triggers_fn2())
