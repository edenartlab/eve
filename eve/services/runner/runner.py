import logging
import os
from eve import db
from pathlib import Path
import modal
import sentry_sdk
from eve.services.runner.runner_tasks import (
    cancel_stuck_tasks,
    download_nsfw_models,
    generate_lora_thumbnails,
    run_nsfw_detection,
)

root_dir = Path(__file__).parent.parent.parent
workflows_dir = root_dir / ".." / "workflows"

app_name = f"runner-{db.lower()}"
logging.basicConfig(level=logging.INFO)

app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE")})
    .apt_install(
        "git",
        "libmagic1",
        "ffmpeg",
        "wget",
        "libnss3",
        "libnspr4",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libatspi2.0-0",
        "libxcomposite1",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .pip_install("numpy<2.0", "torch==2.0.1", "torchvision", "transformers", "Pillow")
    .run_commands(["playwright install"])
    .run_function(download_nsfw_models)
    .copy_local_dir(str(workflows_dir), "/workflows")
)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def cancel_stuck_tasks_fn():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        print(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def run_nsfw_detection_fn():
    try:
        await run_nsfw_detection()
    except Exception as e:
        print(f"Error running nsfw detection: {e}")
        sentry_sdk.capture_exception(e)


@app.function(
    image=image, concurrency_limit=1, schedule=modal.Period(minutes=15), timeout=3600
)
async def generate_lora_thumbnails_fn():
    try:
        await generate_lora_thumbnails()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


# @app.function(
#     image=image, concurrency_limit=15, schedule=modal.Period(minutes=1), timeout=3600
# )
# async def run_twitter_replybots_fn():
#     try:
#         await run_twitter_automation()
#     except Exception as e:
#         print(f"Error running Twitter replybots: {e}")
#         sentry_sdk.capture_exception(e)
