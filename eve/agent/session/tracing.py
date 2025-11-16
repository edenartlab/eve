"""
Distributed tracing utilities for session and trigger lifecycle monitoring.

Uses Sentry for tracing to provide visibility into:
- async_prompt_session lifecycle
- trigger execution flow
- tool execution timing
- memory formation
"""

import functools
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional

import sentry_sdk


def should_trace() -> bool:
    """Determine if tracing should be enabled based on environment."""
    # Trace if Sentry is configured and not explicitly disabled
    sentry_dsn = os.getenv("SENTRY_DSN")
    tracing_disabled = os.getenv("DISABLE_TRACING", "false").lower() == "true"
    return bool(sentry_dsn) and not tracing_disabled


@contextmanager
def trace_operation(operation_name: str, **tags):
    """
    Context manager for tracing a synchronous operation.

    Usage:
        with trace_operation("build_system_message", session_id=session_id):
            # your code here
    """
    if not should_trace():
        yield None
        return

    with sentry_sdk.start_span(op=operation_name) as span:
        # Add tags to span
        for key, value in tags.items():
            if value is not None:
                span.set_tag(key, str(value))
        yield span


@asynccontextmanager
async def trace_async_operation(operation_name: str, **tags):
    """
    Async context manager for tracing an async operation.

    Usage:
        async with trace_async_operation("determine_actors", session_id=session_id):
            # your async code here
    """
    if not should_trace():
        yield None
        return

    with sentry_sdk.start_span(op=operation_name) as span:
        # Add tags to span
        for key, value in tags.items():
            if value is not None:
                span.set_tag(key, str(value))
        yield span


def start_transaction(
    name: str,
    op: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    trigger_id: Optional[str] = None,
    **additional_tags,
) -> Optional[Any]:
    """
    Start a new Sentry transaction for a top-level operation.

    Args:
        name: Transaction name (e.g., "prompt_session", "trigger_execution")
        op: Operation type (e.g., "session", "trigger")
        session_id: Optional session ID for context
        user_id: Optional user ID for context
        agent_id: Optional agent ID for context
        trigger_id: Optional trigger ID for context
        **additional_tags: Any additional tags to attach

    Returns:
        Transaction object or None if tracing is disabled
    """
    if not should_trace():
        return None

    transaction = sentry_sdk.start_transaction(name=name, op=op)

    # Set standard tags
    if session_id:
        transaction.set_tag("session_id", session_id)
    if user_id:
        transaction.set_tag("user_id", user_id)
    if agent_id:
        transaction.set_tag("agent_id", agent_id)
    if trigger_id:
        transaction.set_tag("trigger_id", trigger_id)

    # Set additional tags
    for key, value in additional_tags.items():
        if value is not None:
            transaction.set_tag(key, str(value))

    return transaction


def set_transaction_data(key: str, value: Any):
    """
    Set custom data on the current transaction.

    Args:
        key: Data key
        value: Data value
    """
    if not should_trace():
        return

    scope = sentry_sdk.get_current_scope()
    if scope and scope.transaction:
        scope.transaction.set_data(key, value)


def set_transaction_tag(key: str, value: Any):
    """
    Set a tag on the current transaction.

    Args:
        key: Tag key
        value: Tag value
    """
    if not should_trace():
        return

    scope = sentry_sdk.get_current_scope()
    if scope:
        scope.set_tag(key, str(value))


def trace_function(operation_name: Optional[str] = None, **default_tags):
    """
    Decorator for tracing function calls.

    Usage:
        @trace_function("validate_session", component="session")
        def validate_prompt_session(session, context):
            # your code here
    """

    def decorator(func: Callable):
        op_name = operation_name or f"function.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not should_trace():
                return func(*args, **kwargs)

            with sentry_sdk.start_span(op=op_name) as span:
                for key, value in default_tags.items():
                    span.set_tag(key, str(value))
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_function(operation_name: Optional[str] = None, **default_tags):
    """
    Decorator for tracing async function calls.

    Usage:
        @trace_async_function("determine_actors", component="session")
        async def determine_actors(session, context):
            # your async code here
    """

    def decorator(func: Callable):
        op_name = operation_name or f"function.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not should_trace():
                return await func(*args, **kwargs)

            with sentry_sdk.start_span(op=op_name) as span:
                for key, value in default_tags.items():
                    span.set_tag(key, str(value))
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def add_breadcrumb(
    message: str,
    category: str = "info",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
):
    """
    Add a breadcrumb to the current scope for debugging.

    Args:
        message: Breadcrumb message
        category: Category (e.g., "session", "tool", "memory")
        level: Level (e.g., "info", "warning", "error")
        data: Optional additional data
    """
    if not should_trace():
        return

    sentry_sdk.add_breadcrumb(
        message=message, category=category, level=level, data=data or {}
    )
