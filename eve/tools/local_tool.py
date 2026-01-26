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
        self._cancellation_events = {}  # Track cancellation events per task

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
        # Create a cancellation event for this task
        cancellation_event = asyncio.Event()
        self._cancellation_events[task_id] = cancellation_event
        # Run task with cancellation event (can't use run_task due to decorator)
        background_task = asyncio.create_task(
            self._run_task_with_cancellation(task, cancellation_event)
        )
        self._tasks[task_id] = background_task
        return task_id

    async def _run_task_with_cancellation(
        self, task: Task, cancellation_event: asyncio.Event
    ):
        """Run task handler with cancellation support, bypassing @task_handler_func."""
        from datetime import datetime, timezone

        start_time = datetime.now(timezone.utc)
        queue_time = (start_time - task.createdAt).total_seconds()
        task.update(status="running", performance={"waitTime": queue_time})

        try:
            context = ToolContext(
                args=task.args,
                user=str(task.user) if task.user else None,
                agent=str(task.agent) if task.agent else None,
                session=str(task.session) if task.session else None,
                message=str(task.message) if task.message else None,
                tool_call_id=str(task.tool_call_id) if task.tool_call_id else None,
                cancellation_event=cancellation_event,
            )
            handler = load_handler(self.parent_tool or self.key)
            result = await handler(context)

            # Handle cancelled status from handler
            if result.get("status") == "cancelled":
                task.update(status="cancelled")
                task.refund_manna()
                cancelled_result = {
                    "output": [],
                    "status": "cancelled",
                    "error": result.get("error", "Task was cancelled"),
                }
                return {"status": "cancelled", "result": [cancelled_result]}

            # Process successful result
            from .. import utils

            result["output"] = (
                result["output"]
                if isinstance(result["output"], list)
                else [result["output"]]
            )
            result = utils.upload_result(result, save_thumbnails=True)

            # Preserve status from result, default to completed only if not set
            final_status = result.get("status", "completed")
            result["status"] = final_status

            end_time = datetime.now(timezone.utc)
            run_time = (end_time - start_time).total_seconds()

            # Wrap result in a list to match task_handler_func structure
            results_list = [result]

            # Use the result's status for the task update (not hardcoded "completed")
            task.update(
                status=final_status,
                result=results_list,
                performance={"waitTime": queue_time, "runTime": run_time},
            )
            # Return wrapper structure matching _task_handler
            return {"status": final_status, "result": results_list}

        except asyncio.CancelledError:
            task.update(status="cancelled")
            task.refund_manna()
            cancelled_result = {
                "output": [],
                "status": "cancelled",
                "error": "Task was cancelled by user",
            }
            return {"status": "cancelled", "result": [cancelled_result]}

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.update(status="failed", error=str(e))
            task.refund_manna()
            return {"status": "failed", "error": str(e)}

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
            # Clean up cancellation event
            if task.handler_id in self._cancellation_events:
                del self._cancellation_events[task.handler_id]
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
            # Clean up cancellation event
            if task.handler_id in self._cancellation_events:
                del self._cancellation_events[task.handler_id]
            return {
                "status": "failed",
                "error": f"Task timed out after {LOCAL_TOOL_TIMEOUT} seconds",
            }
        except asyncio.CancelledError:
            # Clean up cancellation event
            if task.handler_id in self._cancellation_events:
                del self._cancellation_events[task.handler_id]
            return {"status": "cancelled"}

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        # Set the cancellation event first (for cooperative cancellation)
        if task.handler_id in self._cancellation_events:
            self._cancellation_events[task.handler_id].set()

        if task.handler_id in self._tasks:
            background_task = self._tasks[task.handler_id]
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
            finally:
                del self._tasks[task.handler_id]
                # Clean up cancellation event
                if task.handler_id in self._cancellation_events:
                    del self._cancellation_events[task.handler_id]


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
    """Legacy run_task for backward compatibility. Does not support cancellation."""
    context = ToolContext(
        args=args,
        user=str(user) if user else None,
        agent=str(agent) if agent else None,
        session=str(session) if session else None,
        message=str(message) if message else None,
        tool_call_id=str(tool_call_id) if tool_call_id else None,
    )
    return await load_handler(tool_key)(context)
