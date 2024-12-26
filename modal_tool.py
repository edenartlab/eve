import modal

from eve.task import task_handler_func
from eve.tools import handlers
from eve import eden_utils

app = modal.App(
    name="handlers3",
    secrets=[
        modal.Secret.from_name("s3-credentials"),
        modal.Secret.from_name("mongo-credentials"),
        modal.Secret.from_name("replicate"),
        modal.Secret.from_name("openai"),
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("elevenlabs"),
        modal.Secret.from_name("hedra"),
        modal.Secret.from_name("newsapi"),
        modal.Secret.from_name("runway"),
        modal.Secret.from_name("sentry"),
    ],   
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagic1", "ffmpeg", "wget")
    .pip_install(
        "httpx", "tqdm", "websocket-client", "gitpython", "boto3", "omegaconf",
        "requests", "Pillow", "fastapi==0.103.1", "python-magic", "replicate", 
        "python-dotenv", "pyyaml", "instructor==1.2.6", "torch==2.3.1", "torchvision", "packaging",
        "torchaudio", "pydub", "moviepy==1.0.3", "accelerate", "pymongo", "google-cloud-aiplatform", 
        "runwayml", "elevenlabs", "sentry-sdk", "blurhash"
    )
    .pip_install("web3<7.6.1", "requests_oauthlib")
)

@app.function(image=image, timeout=3600)
async def run(tool_key: str, args: dict, db: str):
    result = await handlers[tool_key](args, db=db)
    return eden_utils.upload_result(result, db=db)

@app.function(image=image, timeout=3600)
@task_handler_func
async def run_task(tool_key: str, args: dict, db: str):
    return await handlers[tool_key](args, db=db)
