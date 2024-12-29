import os
import modal

from eve.clients.discord.client import start as discord_start

db = os.getenv("DB", "STAGE").upper()
print("THE DB IS", db)
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
app_name = "client-discord-prod" if db == "PROD" else "client-discord-stage"
env_file = ".env" if db == "PROD" else ".env.STAGE"

if os.path.exists(env_file):
    print(env_file)
    print("ENV FILE EXISTS")
else:
    print("ENV FILE DOES NOT EXIST")
    raise Exception(f"ENV FILE DOES NOT EXIST: {env_file}")

app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("client-secrets"),
        modal.Secret.from_name("eve-secrets", environment_name="main"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("py-cord>=2.4.1")
    .copy_local_dir("../workflows", "/workflows")
)


@app.function(image=image, keep_warm=1, concurrency_limit=1, timeout=60 * 60 * 24)
@modal.asgi_app()
def modal_app() -> None:
    discord_start(
        env=env_file,
        db=db,
        local=False,
    )
