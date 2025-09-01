import logging
import os
from pathlib import Path
from eve.api.api_requests import PostingInstructions
import modal
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk

from typing import Optional, Literal
from pydantic import BaseModel

from eve.agent.session.models import Trigger



from eve import auth, db

app_name = f"api2-{db.lower()}"

logging.getLogger("ably").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)




# image = modal.Image.debian_slim().pip_install("fastapi[standard]")
# app = modal.App("eden-service", image=image)





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




# from eve.agent.session.triggers import (
from eve.trigger import (
    handle_trigger_create,
    handle_trigger_stop,
    handle_trigger_delete,
    handle_trigger_get,
    handle_trigger_run,
    # handle_trigger_posting,
    execute_trigger
)


from eve.api.api_requests import (
    CronSchedule,
    UpdateConfig,
)

class CreateTriggerRequest(BaseModel):
    agent: str
    user: str
    name: Optional[str] = "Untitled Task"  # Add name field with default
    instruction: str
    posting_instructions: Optional[PostingInstructions] = None
    think: Optional[bool] = None
    schedule: CronSchedule
    update_config: Optional[UpdateConfig] = None
    session_type: Literal["new", "another"] = "new"
    session: Optional[str] = None

@web_app.post("/triggers/create")
async def trigger_create(
    request: CreateTriggerRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(auth.authenticate_admin),
):
    return await handle_trigger_create(request, background_tasks)


# @web_app.post("/triggers/stop")
# async def trigger_stop(
#     request: DeleteTriggerRequest, 
#     _: dict = Depends(auth.authenticate_admin)
# ):
#     return await handle_trigger_stop(request)


# @web_app.post("/triggers/delete")
# async def trigger_delete(
#     request: DeleteTriggerRequest, 
#     _: dict = Depends(auth.authenticate_admin)
# ):
#     return await handle_trigger_delete(request)


# @web_app.post("/triggers/run")
# async def trigger_run(
#     request: RunTriggerRequest, 
#     _: dict = Depends(auth.authenticate_admin)
# ):
#     return await handle_trigger_run(request)


web_app.get("/triggers/{trigger_id}")(
    handle_trigger_get
)


execute_trigger_fn = app.function(
    image=image, 
    max_containers=4
)(execute_trigger)
# async def execute_trigger_fn(trigger: Trigger):
#     return await execute_trigger(trigger)



@app.function(
    image=image, 
    max_containers=1, 
    schedule=modal.Period(minutes=2), 
    timeout=300
)
async def run_scheduled_triggers():
    current_time = datetime.now(timezone.utc)

    # Find active triggers which should be run now
    triggers = Trigger.find({
        "status": "active",
        "deleted": {"$ne": True},
        "next_scheduled_run": {"$lte": current_time},
    })
    from bson import ObjectId
    triggers = [Trigger.from_mongo(ObjectId(t)) for t in ["68b3bca333da060a73cef02a", "68b3bc8f33da060a73cef029"]]    

    logger.info(f"Running {len(triggers)} triggers")

    # from eve.api.api import execute_trigger_fn

    async for result in execute_trigger_fn.map.aio(triggers):
        print(result)
