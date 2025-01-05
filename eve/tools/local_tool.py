import uuid
from typing import Dict
import asyncio

from ..task import Task, task_handler_func
from ..tool import Tool
from .tool_handlers import handlers


class LocalTool(Tool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tasks = {}

    @Tool.handle_run
    async def async_run(self, args: Dict):
        result = await handlers[self.parent_tool or self.key](args)
        return result
    
    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        task_id = str(uuid.uuid4())
        background_task = asyncio.create_task(run_task(task))
        self._tasks[task_id] = background_task
        return task_id
    
    @Tool.handle_wait
    async def async_wait(self, task: Task):
        if task.handler_id not in self._tasks:
            raise ValueError(f"No task found with id {task.handler_id}")            
        try:
            result = await self._tasks[task.handler_id]
            del self._tasks[task.handler_id]
            return result
        except asyncio.CancelledError:
            return {"status": "cancelled"}
    
    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        if task.handler_id in self._tasks:
            background_task = self._tasks[task.handler_id]
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
            finally:
                del self._tasks[task.handler_id]
        

@task_handler_func
async def run_task(tool_key: str, args: dict):
    return await handlers[tool_key](args)
