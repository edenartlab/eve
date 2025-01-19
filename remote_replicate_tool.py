from pathlib import Path
import os
import modal
import replicate

from eve.tools.replicate_tool import replicate_update_task
from eve.tool import Tool
from eve.task import Task

db = os.getenv("DB", "STAGE").upper()
app_name = f"remote-replicate-{db}"

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
    .run_commands(["playwright install"])
    .env({"DB": db})
)

@app.function(image=image, timeout=3600)
async def run_task(task: Task):
    task.update(status="running")
    tool = Tool.load(task.tool)
    args = tool.prepare_args(task.args)
    args = tool._format_args_for_replicate(args)
    replicate_model = tool._get_replicate_model(task.args)
    output = await replicate.async_run(replicate_model, input=args)
    result = replicate_update_task(task, "succeeded", None, output, "normal")
    return result
