from bson import ObjectId
from typing import Dict, Any, Optional, Literal, List
from functools import wraps
from datetime import datetime, timezone
import asyncio
import traceback

from .user import User
from .mongo import Document, Collection
from . import eden_utils
from . import sentry_sdk



@Collection("creations3")
class Creation(Document):
    user: ObjectId
    requester: ObjectId
    task: ObjectId
    tool: str
    filename: str
    mediaAttributes: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    public: bool = False
    deleted: bool = False

    def __init__(self, **data):
        if isinstance(data.get('user'), str):
            data['user'] = ObjectId(data['user'])
        if isinstance(data.get('requesteder'), str):
            data['requester'] = ObjectId(data['requester'])
        if isinstance(data.get('task'), str):
            data['task'] = ObjectId(data['task'])
        super().__init__(**data)


@Collection("tasks3")
class Task(Document):
    user: ObjectId
    requester: ObjectId
    tool: str
    parent_tool: Optional[str] = None
    output_type: str
    args: Dict[str, Any]
    mock: bool = False
    cost: float = None
    handler_id: Optional[str] = None
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    error: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    performance: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        if isinstance(data.get('user'), str):
            data['user'] = ObjectId(data['user'])
        if isinstance(data.get('requester'), str):
            data['requester'] = ObjectId(data['requester'])
        super().__init__(**data)

    @classmethod
    def from_handler_id(self, handler_id, db):
        tasks = self.get_collection(db)
        task = tasks.find_one({"handler_id": handler_id})
        if not task:
            raise Exception("Task not found")    
        return super().load(self, task["_id"], db)


def task_handler_func(func):
    @wraps(func)
    async def wrapper(task: Task):
        return await _task_handler(func, task)
    return wrapper


def task_handler_method(func):
    @wraps(func)
    async def wrapper(self, task: Task):
        return await _task_handler(func, self, task)
    return wrapper


async def _preprocess_task(task: Task):
    """Helper function that sleeps for 5 seconds"""
    await asyncio.sleep(5)
    return {"name": "this is a tbd side task"}


async def _task_handler(func, *args, **kwargs):
    task = kwargs.pop("task", args[-1])
    start_time = datetime.now(timezone.utc)
    queue_time = (start_time - task.createdAt).total_seconds()

    task.update(
        status="running",
        performance={"waitTime": queue_time}
    )
    
    results = []
    n_samples = task.args.get("n_samples", 1)
    output_type = task.output_type

    try:
        for i in range(n_samples):
            task_args = task.args.copy()
            if "seed" in task_args:
                task_args["seed"] = task_args["seed"] + i

            # Run both functions concurrently
            main_task = func(*args[:-1], task.parent_tool or task.tool, task_args, task.db)
            preprocess_task = _preprocess_task(task)
            result, preprocess_result = await asyncio.gather(main_task, preprocess_task)

            if output_type in ["image", "video", "audio", "lora"]:
                result["output"] = result["output"] if isinstance(result["output"], list) else [result["output"]]
                result = eden_utils.upload_result(result, db=task.db, save_thumbnails=True, save_blurhash=True)

                for output in result["output"]:
                    name = preprocess_result.get("name") or task_args.get("prompt") or args.get("text_input")
                    if not name:
                        name = args.get("interpolation_prompts") or args.get("interpolation_texts")
                        if name:
                            name = " to ".join(name)
                    new_creation = Creation(
                        user=task.user,
                        requester=task.requester,
                        task=task.id,
                        tool=task.tool,
                        filename=output['filename'],
                        mediaAttributes=output['mediaAttributes'],
                        name=name
                    )
                    new_creation.save(db=task.db)
                    output["creation"] = new_creation.id

            results.extend([result])

            if i == n_samples - 1:
                task_update = {
                    "status": "completed", 
                    "result": results
                }

            else:
                task_update = {
                    "status": "running", 
                    "result": results
                }
                task.update(**task_update)

        return task_update.copy()

    except Exception as error:
        sentry_sdk.capture_exception(error)
        print(traceback.format_exc())

        task_update = {
            "status": "failed",
            "error": str(error),
        }
        
        n_samples = task.args.get("n_samples", 1)
        refund_amount = (task.cost or 0) * (n_samples - len(task.result or [])) / n_samples
        user = User.from_mongo(task.user, db=task.db)
        user.refund_manna(refund_amount)
        
        return task_update.copy()

    finally:
        run_time = datetime.now(timezone.utc) - start_time
        task_update["performance"] = {
            "waitTime": queue_time,
            "runTime": run_time.total_seconds()
        }
        task.update(**task_update)
