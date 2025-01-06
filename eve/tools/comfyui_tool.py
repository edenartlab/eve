import modal
import os
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from ..mongo import get_collection
from ..tool import Tool
from ..task import Task


class ComfyUIRemap(BaseModel):
    node_id: int
    field: str
    subfield: str
    map: Dict[str, str]

class ComfyUIInfo(BaseModel):
    node_id: int
    field: str
    subfield: str
    preprocessing: Optional[str] = None
    remap: Optional[List[ComfyUIRemap]] = None

class ComfyUITool(Tool):
    workspace: str
    comfyui_output_node_id: int
    comfyui_intermediate_outputs: Optional[Dict[str, int]] = None
    comfyui_map: Dict[str, ComfyUIInfo] = Field(default_factory=dict)

    @classmethod
    def convert_from_yaml(cls, schema: dict, file_path: str = None) -> dict:
        schema["comfyui_map"] = {}
        for field, props in schema.get('parameters', {}).items():
            if 'comfyui' in props:
                schema["comfyui_map"][field] = props['comfyui']
        schema["workspace"] = schema.get("workspace") or file_path.replace("api.yaml", "test.json").split("/")[-4]
        return super().convert_from_yaml(schema, file_path)
    
    @Tool.handle_run
    async def async_run(self, args: Dict):
        db = os.getenv("DB")
        cls = modal.Cls.lookup(
            f"comfyui-{self.workspace}-{db}", 
            "ComfyUI", 
            environment_name="main"
        )
        result = await cls().run.remote.aio(self.parent_tool or self.key, args)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        db = os.getenv("DB")
        cls = modal.Cls.lookup(
            f"comfyui-{self.workspace}-{db}", 
            "ComfyUI",
            environment_name="main"
        )
        job = await cls().run_task.spawn.aio(task)
        return job.object_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        fc = modal.functions.FunctionCall.from_id(task.handler_id)
        await fc.get.aio()
        task.reload()
        return task.model_dump(include={"status", "error", "result"})
        
    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        fc = modal.functions.FunctionCall.from_id(task.handler_id)
        await fc.cancel.aio()


class ComfyUIToolLegacy(ComfyUITool):
    """For legacy/private workflows"""
    
    @Tool.handle_run
    async def async_run(self, args: Dict):
        db = os.getenv("DB")
        cls = modal.Cls.lookup(
            f"comfyui-{self.key}",
            "ComfyUI", 
            environment_name="main"
        )
        result = await cls().run.remote.aio(
            workflow_name=self.key, 
            args=args,
            env=db
        )
        result = {"output": result}
        print(result)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        # hack to accommodate legacy comfyui tasks
        # 1) copy task to tasks2 collection (rename tool to workflow)
        # 2) spawn new job, env=DB
        db = os.getenv("DB")

        task_data = task.model_dump(by_alias=True)
        task_data["workflow"] = task_data.pop("tool")
        tasks2 = get_collection("tasks2")
        tasks2.insert_one(task_data)

        cls = modal.Cls.lookup(
            f"comfyui-{self.key}",
            "ComfyUI",
            environment_name="main"
        )
        job = await cls().run_task.spawn.aio(
            task_id=ObjectId(task_data["_id"]), 
            env=db
        )
        return job.object_id
    
    @Tool.handle_wait
    async def async_wait(self, task: Task):
        # hack to accommodate legacy comfyui tasks
        # 1) get completed task from tasks2 collection
        # 2) copy over canonical test
        # 3) return new task
        fc = modal.functions.FunctionCall.from_id(task.handler_id)
        await fc.get.aio()
        task.reload()

        tasks2 = get_collection("tasks2")
        task2 = tasks2.find_one({"_id": task.id})
        task.update(
            status=task2["status"],
            error=task2["error"],
            result=task2["result"]
        )

        return task.model_dump(include={"status", "error", "result"})
