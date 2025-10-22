import modal
import os
from typing import Dict

from ..task import Task
from ..tool import Tool, tool_context


@tool_context("modal")
class ModalTool(Tool):
    @Tool.handle_run
    async def async_run(
        self,
        args: Dict,
        user_id: str = None,
        agent_id: str = None,
        session_id: str = None,
    ):
        db = os.getenv("DB", "STAGE").upper()
        func = modal.Function.from_name(
            f"api-{db.lower()}", "run", environment_name="main"
        )
        result = await func.remote.aio(tool_key=self.parent_tool or self.key, args=args)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        db = os.getenv("DB", "STAGE").upper()
        func = modal.Function.from_name(
            f"api-{db.lower()}", "run_task", environment_name="main"
        )
        # job = func.spawn(task, parent_tool=self.parent_tool)
        job = await func.spawn.aio(task)
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
