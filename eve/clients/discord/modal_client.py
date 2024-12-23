import modal

from eve.clients.discord.client import start as discord_start


app = modal.App(
    name="client-discord",
    secrets=[
        modal.Secret.from_name("client-secrets"),
        modal.Secret.from_name("eve-secrets", environment_name="main"),
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
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("py-cord>=2.4.1")
    .copy_local_dir("../workflows", "/workflows")
)


@app.function(image=image, keep_warm=1, concurrency_limit=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app() -> None:
    discord_start(env=".env")
