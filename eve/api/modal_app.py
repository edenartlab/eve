

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




from eve import auth, db

app_name = f"api2-{db.lower()}"

logging.getLogger("ably").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)




# image = modal.Image.debian_slim().pip_install("fastapi[standard]")
# app = modal.App("eden-service", image=image)





# class SentryContextMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request, call_next):
#         with sentry_sdk.configure_scope() as scope:
#             scope.set_tag("package", "eve-api")

#             # Extract client context from headers
#             client_platform = request.headers.get("X-Client-Platform")
#             client_deployment_id = request.headers.get("X-Client-Deployment-Id")
#             if client_platform:
#                 scope.set_tag("client_platform", client_platform)
#             if client_deployment_id:
#                 scope.set_tag("client_deployment_id", client_deployment_id)

#             scope.set_context(
#                 "api",
#                 {
#                     "endpoint": request.url.path,
#                     "modal_serve": os.getenv("MODAL_SERVE"),
#                     "client_platform": client_platform,
#                     "client_deployment_id": client_deployment_id,
#                 },
#             )
#         return await call_next(request)


# web_app = FastAPI()
# web_app.add_middleware(SentryContextMiddleware)
# web_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
# bearer_scheme = HTTPBearer(auto_error=False)
# background_tasks: BackgroundTasks = BackgroundTasks()




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
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .add_local_dir(str(workflows_dir), "/workflows")
    .add_local_file(str(root_dir / "pyproject.toml"), "/eve/pyproject.toml")
    .add_local_python_source("eve", ignore=[])
    .add_local_python_source("api", ignore=[])
)

