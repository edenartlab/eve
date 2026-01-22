import asyncio
import traceback
from typing import Any, Dict, Optional

from loguru import logger
from sentry_sdk import capture_exception

from eve.agent.llm.providers.fake import build_fake_tool_result_payload
from eve.agent.llm.util import is_fake_llm_mode, should_force_fake_response
from eve.agent.session.models import (
    ChatMessage,
    LLMContext,
    Session,
    SessionUpdate,
    ToolCall,
    UpdateType,
)
from eve.agent.session.tracing import add_breadcrumb
from eve.tool import Tool

from .budget import update_session_budget


def create_fake_tool_result(
    tool: Tool,
    tool_call: ToolCall,
    user_id: Optional[str],
    agent_id: Optional[str],
    session_id: Optional[str],
    message_id: Optional[str],
    public: bool,
) -> Dict[str, Any]:
    """Create placeholder task and result data for fake chat mode."""
    args = tool_call.args or {}
    try:
        tool_call.args = tool.prepare_args(args.copy())
    except Exception as exc:
        logger.warning(
            f"Failed to normalize args for fake tool call {tool_call.tool}: {exc}"
        )

    result_payload = build_fake_tool_result_payload(
        tool_call.tool, getattr(tool, "name", None)
    )

    return {
        "status": "completed",
        "result": result_payload,
        "cost": 0,
        "task": None,
    }


async def async_run_tool_call_with_cancellation(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
    cancellation_event: asyncio.Event = None,
):
    """
    Cancellation-aware version of async_run_tool_call that can be interrupted
    """

    if tool_call.tool == "web_search":
        return tool_call.result

    tool = llm_context.tools.get(tool_call.tool)
    if not tool and tool_call.tool.startswith("tool_"):
        stripped_tool = tool_call.tool[len("tool_") :]
        tool = llm_context.tools.get(stripped_tool)
        if tool:
            logger.warning(
                f"Mapped tool call name from {tool_call.tool} to {stripped_tool}"
            )
            tool_call.tool = stripped_tool

    # Fallback: try loading tool directly if not in llm_context
    # This handles cases where hot-reload invalidated the tools dictionary
    if not tool:
        try:
            tool = Tool.load(key=tool_call.tool)
            if tool:
                logger.warning(
                    f"Tool {tool_call.tool} not in llm_context.tools, loaded directly as fallback"
                )
        except Exception as e:
            logger.warning(f"Failed to load tool {tool_call.tool} as fallback: {e}")

    if not tool:
        raise KeyError(tool_call.tool)

    if is_fake_llm_mode() or should_force_fake_response(llm_context):
        return create_fake_tool_result(
            tool,
            tool_call,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            message_id=message_id,
            public=public,
        )

    # Start the task
    try:
        task = await tool.async_start_task(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            message_id=message_id,
            tool_call_id=tool_call.id,
            args=tool_call.args,
            mock=False,
            public=public,
            is_client_platform=is_client_platform,
        )
    except Exception as e:
        # If task creation fails, return error without task reference
        return {
            "status": "failed",
            "error": f"Failed to start task: {str(e)}",
            "cost": 0,
            "task": None,
        }

    # If no cancellation event, fall back to normal behavior
    if not cancellation_event:
        try:
            result = await tool.async_wait(task)
        except Exception as e:
            # If task waiting fails, return error with task reference
            return {
                "status": "failed",
                "error": str(e),
                "cost": getattr(task, "cost", 0),
                "task": getattr(task, "id", None),
            }
    else:
        # Race between task completion and cancellation
        wait_task = asyncio.create_task(tool.async_wait(task))

        try:
            # Wait for either task completion or cancellation
            done, pending = await asyncio.wait(
                [wait_task, asyncio.create_task(cancellation_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks
            for task_obj in pending:
                task_obj.cancel()
                try:
                    await task_obj
                except asyncio.CancelledError:
                    pass

            # Check if cancellation happened first
            if cancellation_event.is_set():
                # Try to cancel the task
                try:
                    await tool.async_cancel(task)
                except Exception as e:
                    logger.error(f"Failed to cancel task {task.id}: {e}")

                return {
                    "status": "cancelled",
                    "error": "Task cancelled by user",
                    "cost": getattr(task, "cost", 0),
                    "task": getattr(task, "id", None),
                }
            else:
                # Task completed normally
                result = wait_task.result()

        except Exception as e:
            # If anything goes wrong, try to cancel the task
            try:
                if not wait_task.done():
                    wait_task.cancel()
                    await tool.async_cancel(task)
            except Exception:
                pass
            # Return failed result with task reference instead of raising
            return {
                "status": "failed",
                "error": str(e),
                "cost": getattr(task, "cost", 0),
                "task": getattr(task, "id", None),
            }

    # Add task.cost, task.id, and status to the result object
    if isinstance(result, dict):
        result["cost"] = getattr(task, "cost", None)
        result["task"] = getattr(task, "id", None)
        # Ensure status is set for successful completion
        if "status" not in result:
            result["status"] = "completed"

    return result


async def process_tool_call(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    tool_call: ToolCall,
    tool_call_index: int,
    cancellation_event: asyncio.Event = None,
    tool_cancellation_event: asyncio.Event = None,
    is_client_platform: bool = False,
    session_run_id: str = None,
):
    # Add breadcrumb for tool execution
    add_breadcrumb(
        f"Processing tool: {tool_call.tool}",
        category="tool",
        data={"tool_name": tool_call.tool, "tool_index": tool_call_index},
    )

    # Update the tool call status to running
    tool_call.status = "running"

    # Update assistant message
    if assistant_message.tool_calls and tool_call_index < len(
        assistant_message.tool_calls
    ):
        assistant_message.update_tool_call(tool_call_index, status="running")

    try:
        # Check for cancellation before starting tool execution
        if (cancellation_event and cancellation_event.is_set()) or (
            tool_cancellation_event and tool_cancellation_event.is_set()
        ):
            tool_call.status = "cancelled"
            cancelled_result = [
                {"status": "cancelled", "message": "Task cancelled by user"}
            ]
            tool_call.result = cancelled_result
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.update_tool_call(
                    tool_call_index, status="cancelled", result=cancelled_result
                )
            return SessionUpdate(
                type=UpdateType.TOOL_CANCELLED,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result={"status": "cancelled", "result": cancelled_result},
            )

        # Use cancellation-aware tool execution
        # Use tool-specific cancellation event if available, otherwise use general cancellation event
        effective_cancellation_event = tool_cancellation_event or cancellation_event
        result = await async_run_tool_call_with_cancellation(
            llm_context,
            tool_call,
            user_id=llm_context.metadata.trace_metadata.user_id
            or llm_context.metadata.trace_metadata.agent_id,
            agent_id=llm_context.metadata.trace_metadata.agent_id,
            session_id=str(session.id),
            message_id=str(assistant_message.id),
            cancellation_event=effective_cancellation_event,
            is_client_platform=is_client_platform,
        )

        # Check for cancellation after tool execution completes
        if cancellation_event and cancellation_event.is_set():
            tool_call.status = "cancelled"
            cancelled_result = [
                {"status": "cancelled", "message": "Task cancelled by user"}
            ]
            tool_call.result = cancelled_result
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.update_tool_call(
                    tool_call_index, status="cancelled", result=cancelled_result
                )
            return SessionUpdate(
                type=UpdateType.TOOL_CANCELLED,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result={"status": "cancelled", "result": cancelled_result},
            )

        # Update the original tool call with result
        if result["status"] == "completed":
            tool_call.status = "completed"
            tool_call.result = result.get("result", [])

            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.update_tool_call(
                    tool_call_index,
                    status="completed",
                    result=tool_call.result,
                    cost=result.get("cost", 0),
                    task=result.get("task"),
                )

            update_session_budget(session, manna_spent=result.get("cost", 0))

            return SessionUpdate(
                type=UpdateType.TOOL_COMPLETE,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result=result,
                session_run_id=session_run_id,
            )

        elif result["status"] == "cancelled":
            tool_call.status = "cancelled"
            # Include the result from the handler if available, otherwise use default
            cancelled_result = result.get("result") or [
                {"status": "cancelled", "message": "Task cancelled by user"}
            ]
            tool_call.result = cancelled_result

            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.update_tool_call(
                    tool_call_index,
                    status="cancelled",
                    result=cancelled_result,
                    cost=result.get("cost", 0),
                    task=result.get("task"),
                )

            return SessionUpdate(
                type=UpdateType.TOOL_CANCELLED,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result={"status": "cancelled", "result": cancelled_result},
            )

        else:
            tool_call.status = "failed"

            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.update_tool_call(
                    tool_call_index,
                    status="failed",
                    error=result.get("error"),
                    cost=result.get("cost", 0),
                    task=result.get("task"),
                )

            return SessionUpdate(
                type=UpdateType.ERROR,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                error=result.get("error"),
                session_run_id=session_run_id,
            )
    except Exception as e:
        capture_exception(e)
        traceback.print_exc()

        # Update the original tool call with error
        tool_call.status = "failed"

        if assistant_message.tool_calls and tool_call_index < len(
            assistant_message.tool_calls
        ):
            assistant_message.update_tool_call(
                tool_call_index,
                status="failed",
                error=str(e),
            )

        return SessionUpdate(
            type=UpdateType.ERROR,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            error=str(e),
            session_run_id=session_run_id,
        )


async def process_tool_calls(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    cancellation_event: asyncio.Event = None,
    tool_cancellation_events: dict = None,
    is_client_platform: bool = False,
    session_run_id: str = None,
):
    tool_calls = assistant_message.tool_calls
    if tool_cancellation_events is None:
        tool_cancellation_events = {}

    for b in range(0, len(tool_calls), 4):
        batch = enumerate(tool_calls[b : b + 4])
        tasks = []
        for idx, tool_call in batch:
            # Create a cancellation event for this specific tool call if not exists
            tool_call_id = tool_call.id
            if tool_call_id not in tool_cancellation_events:
                tool_cancellation_events[tool_call_id] = asyncio.Event()

            # Skip tool calls that are already completed or cancelled
            if tool_call.status in ("completed", "cancelled"):
                continue

            tasks.append(
                process_tool_call(
                    session,
                    assistant_message,
                    llm_context,
                    tool_call,
                    b + idx,
                    cancellation_event,
                    tool_cancellation_events[tool_call_id],
                    is_client_platform,
                    session_run_id,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            yield result
