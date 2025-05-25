import modal
import os

from eve import db
from eve.clients.twitter.client import start as twitter_start


app = modal.App(
    name=f"client-twitter-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
        modal.Secret.from_name("client-secrets"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("requests-oauthlib>=1.3.1")
    .copy_local_dir("../workflows", "/workflows")
)

@app.function(
    image=image,
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def modal_app() -> None:
    twitter_start(env=".env")
