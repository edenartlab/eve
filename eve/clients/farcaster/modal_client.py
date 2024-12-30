import modal
import os
from eve.clients.farcaster.client import create_app
from eve.clients.common import modal_secrets

db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")

app = modal.App(
    name="client-farcaster",
    secrets=modal_secrets(db),
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .apt_install("libmagic1")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("farcaster>=0.7.11")
    .copy_local_dir("../workflows", "/workflows")
)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=1,
)
@modal.asgi_app()
def fastapi_app():
    return create_app(env=".env")
