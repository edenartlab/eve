import modal
from eve import db
from eve.clients.discord.client import create_discord_app


app = modal.App(
    name=f"client-discord-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("../workflows", "/workflows")
    .env({"DB": db})
    .env({"AGENT_ID": ""})
    .env({"CLIENT_DISCORD_TOKEN": ""})
)


@app.function(image=image, min_containers=1, max_containers=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app():
    return create_discord_app()
