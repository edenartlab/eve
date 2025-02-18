import modal

from eve import db
from eve.clients.telegram.client import start

app = modal.App(
    name=f"client-telegram-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("pyproject.toml")
    .run_commands(["playwright install"])
    .copy_local_dir("../workflows", "/workflows")
    .env({"DB": db})
    .env({"AGENT_ID": ""})
    .env({"CLIENT_TELEGRAM_TOKEN": ""})
)


@app.function(image=image, keep_warm=1, concurrency_limit=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app():
    return start(env=".env", local=False)
