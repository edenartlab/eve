from contextvars import ContextVar
from typing import Optional, Tuple
from contextlib import asynccontextmanager

# Context variable to allow handlers to update their parent tool call
_current_tool_call_context: ContextVar[Optional[Tuple]] = ContextVar(
    '_current_tool_call_context',
    default=None
)


@asynccontextmanager
async def tool_call_context(tool_call, assistant_message, tool_call_index):
    """
    Context manager that makes the current tool call accessible to handlers.
    Automatically cleans up on exit.

    This allows handlers (particularly session_post) to update the tool call
    that invoked them, without requiring changes to handler signatures.

    Args:
        tool_call: The ToolCall object being executed
        assistant_message: The ChatMessage containing the tool call
        tool_call_index: Index of this tool call in the message's tool_calls list
    """
    token = _current_tool_call_context.set((tool_call, assistant_message, tool_call_index))
    try:
        yield
    finally:
        _current_tool_call_context.reset(token)


def get_current_tool_call():
    """
    Get the current tool call context if available.

    Returns:
        Tuple of (tool_call, assistant_message, tool_call_index) if within a tool call context,
        None otherwise.
    """
    return _current_tool_call_context.get()
