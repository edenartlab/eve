import modal
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

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
            # f"comfyui-{self.workspace}-{task.db}", 
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

