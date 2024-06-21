"""
- check if user has manna + withdraw
- updating mongo
- if job fails, refund manna
- send back current generators + update
"""

from bson import ObjectId
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from starlette.websockets import WebSocketDisconnect, WebSocketState
import modal
import replicate

from mongo import MongoBaseModel, tasks, models, users, threads
from thread import Thread, UserMessage, get_thread
import auth
import s3
import tools


# Todo: make names unique (case-insensitive)
class Model(MongoBaseModel):
    name: str
    user: ObjectId
    slug: str = None
    args: Dict[str, Any]
    public: bool = False
    checkpoint: str
    thumbnail: str

    def __init__(self, **data):
        super().__init__(**data)
        self.make_slug()

    def make_slug(self):
        name = self.name.lower().replace(" ", "-")
        version = 1 + models.count_documents({"name": self.name, "user": self.user}) 
        username = users.find_one({"_id": self.user})["username"]
        self.slug = f"{username}/{name}/v{version}"

    def save(self):
        super().save(self, models)


class Task(MongoBaseModel):
    workflow: str
    args: Dict[str, Any]
    status: str = "pending"
    error: Optional[str] = None
    result: Optional[Any] = None
    user: ObjectId

    def save(self):
        super().save(self, tasks)
    

async def create(data, user):
    task = Task(**data, user=user["_id"])
    tool = tools.load_tool(task.workflow, f"../workflows/{task.workflow}/api.yaml")
    
    # todo: should call tool.execute(task.args) instead 
    task.args = tools.prepare_args(tool, task.args)
    task.save()
    print("THE ARGS", task.args)
    cls = modal.Cls.lookup("comfyui", task.workflow)
    result = await cls().api.remote.aio(task.args)
    
    if 'error' in result:
        task.status = "failed"
        task.error = str(result['error'])
    else:
        task.status = "completed"
        task.result = result
    
    task.save()
    yield task.model_dump_json()
    

async def train(args: Dict, user):
    tool = tools.load_tool("lora_trainer", f"tools/lora_trainer/api.yaml")
    task = Task(
        workflow="lora_trainer",
        args=args,
        user=user["_id"]
    )
    task.args = tools.prepare_args(tool, task.args)
    task.save()

    args = task.args.copy()
    args['lora_training_urls'] = "|".join(args['lora_training_urls'])

    print("THE ARGS", args)


    deployment = replicate.deployments.get("edenartlab/lora-trainer")
    prediction = deployment.predictions.create(args)
    # deployment = replicate.deployments.get("edenartlab/lora-trainer")
    # prediction = deployment.predictions.create(
    #     input={
    #         "name": name,
    #         "lora_training_urls": "|".join(args.training_images), 
    #         # "ti_lr": 0.001,
    #         # "unet_lr": 0.001,
    #         # "n_tokens": 2,
    #         # "use_dora": False,
    #         # "lora_rank": 16,
    #         # "resolution": 512,
    #         "concept_mode": args.mode,
    #         # "max_train_steps": 400,
    #         "sd_model_version": args.base_model,
    #         # "train_batch_size": 4
    #     }
    # )

    prediction.wait()
    output = prediction.output[-1]
    
    if not output.get('files'):
        task.status = "failed"
        task.error = "No files found in output"
        task.save()
        raise Exception("No files found in output")
    
    tarfile = output['files'][0]
    thumbnail = output['thumbnails'][0]
    
    tarfile_url = s3.upload_file_from_url(tarfile)
    thumbnail_url = s3.upload_file_from_url(thumbnail)
    
    task.result = output
    task.status = "completed"
    task.save()

    model = Model(
        name=args["name"],
        user=user["_id"],
        args=task.args,
        checkpoint=tarfile_url, 
        thumbnail=thumbnail_url
    )

    model.save()
    yield model.model_dump_json()


class ChatRequest(BaseModel):
    message: UserMessage
    thread_id: Optional[str] = None

async def chat(data, user):
    request = ChatRequest(**data)

    if request.thread_id:
        thread = threads.find_one({"_id": ObjectId(request.thread_id)})
        if not thread:
            raise Exception("Thread not found")
        thread = Thread(**thread)
    else:
        thread = Thread()

    async for response in thread.prompt(request.message):
        yield {
            "message": response.model_dump_json()
        }


async def get_or_create_thread(request: dict, user: dict = Depends(auth.authenticate)):
    print("the request", request)
    # return {"thread_id": "adsfadf"}
    thread_name = request.get("name")
    if not thread_name:
        raise HTTPException(status_code=400, detail="Thread name is required")
    thread = get_thread(thread_name, create_if_missing=True)
    return {"thread_id": str(thread.id)}


def create_handler(task_handler):
    async def websocket_handler(
        websocket: WebSocket, 
        user: dict = Depends(auth.authenticate_ws)
    ):
        await websocket.accept()
        try:
            async for data in websocket.iter_json():
                try:
                    async for response in task_handler(data, user):
                        await websocket.send_json(response)
                    break
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    break
        except WebSocketDisconnect:
            print("WebSocket disconnected by client")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            if websocket.application_state == WebSocketState.CONNECTED:
                print("Closing WebSocket...")
                await websocket.close()
    return websocket_handler


web_app = FastAPI()

web_app.websocket("/ws/create")(create_handler(create))
web_app.websocket("/ws/chat")(create_handler(chat))
web_app.websocket("/ws/train")(create_handler(train))

web_app.post("/thread/create")(get_or_create_thread)


app = modal.App(
    name="tasks",
    secrets=[
        modal.Secret.from_name("s3-credentials"),
        modal.Secret.from_name("clerk-credentials"),
        modal.Secret.from_name("mongo-credentials"),
        modal.Secret.from_name("openai"),
        modal.Secret.from_name("replicate"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libmagic1")
    .pip_install("pyjwt", "httpx", "cryptography", "pymongo", "instructor==1.2.6", 
                 "fastapi==0.103.1", "requests", "pyyaml", "python-dotenv", 
                 "python-socketio", "replicate", "boto3", "python-magic", "Pillow")
    .copy_local_dir("../workflows", remote_path="/workflows")
    .copy_local_dir("tools", remote_path="/root/tools")
)

@app.function(
    image=image, 
    keep_warm=1,
    concurrency_limit=5,
    timeout=1800,
    container_idle_timeout=30,
)
@modal.asgi_app()
def fastapi_app():
    return web_app
