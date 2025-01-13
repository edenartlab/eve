import os
from pathlib import Path
import modal

from eve.task import task_handler_func
from eve import eden_utils
from eve.tools.tool_handlers import handlers

db = os.getenv("DB", "STAGE").upper()
app_name = f"modal-tools-{db.lower()}"

app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

root_dir = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
)

@app.function(image=image, timeout=3600)
async def run(tool_key: str, args: dict):
    result = await handlers[tool_key](args)
    return eden_utils.upload_result(result)

@app.function(image=image, timeout=3600)
@task_handler_func
async def run_task(tool_key: str, args: dict):
    return await handlers[tool_key](args)
