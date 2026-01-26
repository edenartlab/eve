import os

import modal

from ..task import Task
from ..tool import Tool, ToolContext, tool_context

# Tools that need extended timeout (3 hours instead of 1 hour)
LONG_TIMEOUT_TOOLS = {"reel"}


@tool_context("modal")
class ModalTool(Tool):
    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        db = os.getenv("DB", "STAGE").upper()
        tool_key = self.parent_tool or self.key
        func_name = "run_3h" if tool_key in LONG_TIMEOUT_TOOLS else "run"
        func = modal.Function.from_name(
            f"api-{db.lower()}", func_name, environment_name="main"
        )
        result = await func.remote.aio(
            tool_key=tool_key,
            args=context.args,
            user=context.user,
            agent=context.agent,
            session=context.session,
            message=context.message,
            tool_call_id=context.tool_call_id,
        )
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
        # Mark task as cancelled in DB first
        task.update(status="cancelled")

        # Then try to cancel the Modal function
        try:
            fc = modal.functions.FunctionCall.from_id(task.handler_id)
            await fc.cancel.aio()
        except Exception as e:
            # Modal cancellation might fail, but task is already marked cancelled
            from loguru import logger

            logger.warning(f"Failed to cancel Modal function {task.handler_id}: {e}")
