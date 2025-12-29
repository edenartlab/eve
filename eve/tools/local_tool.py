import asyncio
import uuid

from loguru import logger

from ..task import Task, task_handler_func
from ..tool import Tool, ToolContext, tool_context
from .tool_handlers import load_handler

# Timeout for local tool execution (1 hour)
LOCAL_TOOL_TIMEOUT = 3600


@tool_context("local")
class LocalTool(Tool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tasks = {}

    def pre_generate_handler_id(self) -> str:
        """Pre-generate handler_id to avoid extra DB write after task creation."""
        return str(uuid.uuid4())

    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        handler = load_handler(self.parent_tool or self.key)
        result = await handler(context)
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        # Use pre-generated handler_id if available, otherwise generate new one
        task_id = task.handler_id or str(uuid.uuid4())
        background_task = asyncio.create_task(run_task(task))
        self._tasks[task_id] = background_task
        return task_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        if task.handler_id not in self._tasks:
            raise ValueError(f"No task found with id {task.handler_id}")
        try:
            # Apply timeout to prevent runaway local tasks
            result = await asyncio.wait_for(
                self._tasks[task.handler_id], timeout=LOCAL_TOOL_TIMEOUT
            )
            del self._tasks[task.handler_id]
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"Local tool execution timed out after {LOCAL_TOOL_TIMEOUT}s: {task.tool}"
            )
            # Cancel the background task
            if task.handler_id in self._tasks:
                self._tasks[task.handler_id].cancel()
                try:
                    await self._tasks[task.handler_id]
                except asyncio.CancelledError:
                    pass
                del self._tasks[task.handler_id]
            return {
                "status": "failed",
                "error": f"Task timed out after {LOCAL_TOOL_TIMEOUT} seconds",
            }
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
async def run_task(
    tool_key: str,
    args: dict,
    user: str = None,
    agent: str = None,
    session: str = None,
    message: str = None,
    tool_call_id: str = None,
):
    context = ToolContext(
        args=args,
        user=str(user) if user else None,
        agent=str(agent) if agent else None,
        session=str(session) if session else None,
        message=str(message) if message else None,
        tool_call_id=str(tool_call_id) if tool_call_id else None,
    )
    return await load_handler(tool_key)(context)
