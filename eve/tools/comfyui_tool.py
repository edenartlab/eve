import os
import modal
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from ..mongo import get_collection
from ..tool import Tool, tool_context
from ..task import Task
from ..user import User


@tool_context("comfyui")
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
    def handle_exception_routing(cls, user: User):
        if str(user.id) == "66867ea4056de0f554a34a77":
            return "ComfyUITempleAbyss"
        return None

    @classmethod
    def convert_from_yaml(cls, schema: dict, file_path: str = None) -> dict:
        schema["comfyui_map"] = {}
        for field, props in schema.get("parameters", {}).items():
            if "comfyui" in props:
                schema["comfyui_map"][field] = props["comfyui"]
        schema["workspace"] = (
            schema.get("workspace")
            or file_path.replace("api.yaml", "test.json").split("/")[-4]
        )
        return super().convert_from_yaml(schema, file_path)

    @Tool.handle_run
    async def async_run(
        self,
        args: Dict,
        user_id: str = None,
        agent_id: str = None,
        session_id: str = None,
    ):
        db = os.getenv("DB", "STAGE")
        print(f"ComfyUI: comfyui-{self.workspace}-{db}")
        cls = modal.Cls.from_name(
            f"comfyui-{self.workspace}-{db}", "ComfyUIBasic", environment_name="main"
        )
        result = await cls().run.remote.aio(self.parent_tool or self.key, args)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        user = User.from_mongo(task.user)
        special_class = self.handle_exception_routing(user)
        is_subscriber = user.subscriptionTier and user.subscriptionTier > 0
        normal_class = "ComfyUIPremium" if is_subscriber else "ComfyUIBasic"
        modal_class = special_class or normal_class

        db = os.getenv("DB", "STAGE")
        cls = modal.Cls.from_name(
            f"comfyui-{self.workspace}-{db}", modal_class, environment_name="main"
        )
        print(f"comfyui-{self.workspace}-{db}", cls)
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
