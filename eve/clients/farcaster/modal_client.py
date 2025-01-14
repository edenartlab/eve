import modal
import os
from eve import db
from eve.clients.farcaster.client import create_app


app = modal.App(
    name=f"client-farcaster-{db}",
    secrets=[
        modal.Secret.from_name("client-secrets"),
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
    ],
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
