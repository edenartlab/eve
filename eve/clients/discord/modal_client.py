import os
import modal
from eve.clients.discord.client import start as discord_start
from eve.clients.common import modal_secrets

db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")


app = modal.App(
    name="client-discord",
    secrets=modal_secrets(db),
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("pyproject.toml")
    .copy_local_dir("../workflows", "/workflows")
)


@app.function(image=image, keep_warm=1, concurrency_limit=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app() -> None:
    discord_start(env=".env")
