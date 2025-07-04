import modal
import os
from pathlib import Path
from eve import db
from eve.clients.farcaster.client import create_app

root_dir = Path(__file__).parent.parent.parent.parent

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
    .apt_install("libmagic1")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("farcaster>=0.7.11")
    .add_local_dir("../workflows", "/workflows")
    .env({"DB": db})
    .add_local_file(str(root_dir / "pyproject.toml"), "/pyproject.toml")
)


@app.function(
    image=image,
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    return create_app(env=".env")
