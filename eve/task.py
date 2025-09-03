import asyncio
import traceback
from bson import ObjectId
from typing import Dict, Any, Optional, Literal, List
from functools import wraps
from datetime import datetime, timezone

from .user import Manna, Transaction
from .mongo import Document, Collection
from .models import Model
from . import utils
import sentry_sdk


# A list of tools that output media but do not result in new Creations
from .tool_constants import (
    EDEN_DB_TOOLS,
    SOCIAL_MEDIA_TOOLS,
    CONTEXT7_MCP_TOOLS,
    CALCULATOR_MCP_TOOLS
)
NON_CREATION_TOOLS = [
    *EDEN_DB_TOOLS,
    *SOCIAL_MEDIA_TOOLS,
    *CONTEXT7_MCP_TOOLS,
    *CALCULATOR_MCP_TOOLS,
]


@Collection("creations3")
class Creation(Document):
    user: ObjectId
    agent: Optional[ObjectId] = None
    task: ObjectId
    tool: str
    filename: str
    mediaAttributes: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    public: bool = True
    deleted: bool = False

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        if isinstance(data.get("task"), str):
            data["task"] = ObjectId(data["task"])
        super().__init__(**data)


@Collection("collections3")
class CreationsCollection(Document):
    user: ObjectId
    name: str
    creations: Optional[List[ObjectId]] = []
    contributors: Optional[List[ObjectId]] = []
    description: Optional[str] = None
    deleted: bool = False
    public: bool = True
    coverCreation: Optional[ObjectId] = None

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        data["creations"] = [
            ObjectId(creation) if isinstance(creation, str) else creation
            for creation in data.get("creations", [])
        ]
        data["contributors"] = [
            ObjectId(contributor) if isinstance(contributor, str) else contributor
            for contributor in data.get("contributors", [])
        ]
        super().__init__(**data)

    @classmethod
    def load(cls, name, user, create_if_missing=False):
        document = cls.get_collection().find_one({"name": name, "user": user})
        if not document:
            if create_if_missing:
                document = cls(name=name, user=user)
                document.save()
            else:
                raise Exception("Collection not found")
        return cls(**document)

    def add_creation(self, creation: ObjectId):
        creation = ObjectId(creation) if isinstance(creation, str) else creation
        if creation not in self.creations:
            self.creations.append(creation)
        if len(self.creations) == 1:
            self.coverCreation = self.creations[0]
        self.save()

    def remove_creation(self, creation: ObjectId):
        if creation in self.creations:
            self.creations.remove(creation)
        self.save()


@Collection("tasks3")
class Task(Document):
    user: ObjectId
    agent: Optional[ObjectId] = None
    thread: Optional[ObjectId] = None
    tool: str
    parent_tool: Optional[str] = None
    output_type: str
    args: Dict[str, Any]
    mock: bool = False
    cost: float = None
    handler_id: Optional[str] = None
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = (
        "pending"
    )
    public: bool = False
    error: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    performance: Optional[Dict[str, Any]] = {}
    paying_user: Optional[ObjectId] = None

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        super().__init__(**data)

    @classmethod
    def from_handler_id(cls, handler_id):
        tasks = cls.get_collection()
        task = tasks.find_one({"handler_id": handler_id})
        if not task:
            raise Exception("Task not found")
        return cls.from_mongo(task["_id"])

    def spend_manna(self):
        if self.cost == 0:
            return
        manna = Manna.load(self.paying_user or self.user)
        manna.spend(self.cost)
        Transaction(
            manna=manna.id,
            task=self.id,
            amount=-self.cost,
            type="spend",
        ).save()

    def refund_manna(self):
        n_samples = self.args.get("n_samples", 1)
        refund_amount = (
            (self.cost or 0) * (n_samples - len(self.result or [])) / n_samples
        )
        manna = Manna.load(self.paying_user or self.user)
        manna.refund(refund_amount)
        Transaction(
            manna=manna.id,
            task=self.id,
            amount=refund_amount,
            type="refund",
        ).save()


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


# this is not used yet, but can be used for moderating requests
async def _preprocess_task(task: Task):
    """Helper function that sleeps for 5 seconds"""
    await asyncio.sleep(1)
    return {"name": "this is a tbd side task"}


async def _task_handler(func, *args, **kwargs):
    task = kwargs.pop("task", args[-1])
    start_time = datetime.now(timezone.utc)
    queue_time = (start_time - task.createdAt).total_seconds()

    task.update(status="running", performance={"waitTime": queue_time})

    results = []
    task_update = {}
    n_samples = task.args.get("n_samples", 1)
    output_type = task.output_type
    is_creation_tool = not task.tool in NON_CREATION_TOOLS

    try:
        for i in range(n_samples):
            task_args = task.args.copy()
            if "seed" in task_args:
                task_args["seed"] = task_args["seed"] + i

            # Run both functions concurrently
            main_task = func(
                *args[:-1],
                task.parent_tool or task.tool,
                task_args,
                user=task.user,
                agent=task.agent,
            )
            preprocess_task = _preprocess_task(task)

            # preprocess_task is just a stub. it will allow us to parallelize pre-processing tasks that dont want to hold up the main task
            result, _ = await asyncio.gather(main_task, preprocess_task)

            if output_type in ["image", "video", "audio", "lora"] and is_creation_tool:
                result["output"] = (
                    result["output"]
                    if isinstance(result["output"], list)
                    else [result["output"]]
                )
                result = utils.upload_result(
                    result, save_thumbnails=True, save_blurhash=True
                )

                for output in result["output"]:
                    filename = output.get("filename")
                    media_attributes = output.get("mediaAttributes")

                    # Skip if the tool is a non-creation tool
                    if not filename:
                        continue

                    # name = preprocess_result.get("name") or task_args.get("prompt") or args.get("text_input")
                    name = task_args.get("prompt") or task_args.get("text_input")
                    if not name:
                        name = task_args.get("interpolation_prompts") or task_args.get(
                            "interpolation_texts"
                        )
                        if name:
                            name = " to ".join(name)

                    new_creation = Creation(
                        user=task.user,
                        agent=task.agent,
                        task=task.id,
                        tool=task.tool,
                        filename=filename,
                        mediaAttributes=media_attributes,
                        name=name,
                        public=task.public,
                    )
                    new_creation.save()
                    output["creation"] = new_creation.id

                    # increment creation count
                    # if task_args.get("lora"):
                    #     model = Model.from_mongo(task_args.get("lora"))
                    #     model.creationCount += 1
                    #     model.save()

            results.extend([result])

            if i == n_samples - 1:
                task_update = {"status": "completed", "result": results}

            else:
                task_update = {"status": "running", "result": results}
                task.update(**task_update)

        return task_update.copy()

    except Exception as error:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("task_failure", "true")
            scope.set_tag("task_tool_key", task.tool)
            if task.parent_tool:
                scope.set_tag("task_parent_tool", task.parent_tool)
            scope.set_context("task_failure", {
                "task_id": str(task.id),
                "tool": task.tool,
                "parent_tool": task.parent_tool,
                "user": str(task.user),
                "agent": str(task.agent) if task.agent else None,
                "args_keys": list(task.args.keys()) if task.args else [],
                "n_samples": task.args.get("n_samples", 1),
                "output_type": task.output_type,
            })
        sentry_sdk.capture_exception(error)
        print(traceback.format_exc())

        task_update = {
            "status": "failed",
            "error": str(error),
        }
        task.refund_manna()

        return task_update.copy()

    finally:
        run_time = datetime.now(timezone.utc) - start_time
        task_update["performance"] = {
            "waitTime": queue_time,
            "runTime": run_time.total_seconds(),
        }
        task.update(**task_update)
