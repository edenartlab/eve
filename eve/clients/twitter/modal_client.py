import modal

from eve.clients.twitter.client import start as twitter_start

app = modal.App(
    name="client-twitter",
    secrets=[modal.Secret.from_name(s) for s in ["eve-secrets", "client-secrets"]],
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
    .pip_install("requests-oauthlib>=1.3.1")
)


@app.function(
    image=image,
    keep_warm=1,
    concurrency_limit=1,
)
@modal.asgi_app()
def modal_app() -> None:
    twitter_start(env=".env")
