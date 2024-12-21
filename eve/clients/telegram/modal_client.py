import modal

from eve.clients.telegram.client import start as telegram_start

app = modal.App(
    name="client-telegram",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name("verdelis-client-secrets"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "DB": "STAGE",
        }
    )
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("/Users/jmill/projects/eden/eve/pyproject.toml")
    .pip_install("python-telegram-bot>=21.7")
    .copy_local_dir("../workflows", "/workflows")
)


@app.function(image=image, keep_warm=1, concurrency_limit=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app() -> None:
    telegram_start(env=".env")
