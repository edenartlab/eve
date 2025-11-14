from typing import Tuple

from bson import ObjectId
from fastapi import BackgroundTasks

from eve.agent.session.debug_logger import SessionDebugger
from eve.agent.session.models import (
    NotificationConfig,
    PromptSessionContext,
    Session,
    UpdateType,
)
from eve.agent.session_new.runtime import _run_prompt_session_internal
from eve.api.api_requests import PromptSessionRequest
from eve.api.helpers import emit_update
from loguru import logger
import traceback

from .setup import setup_session


def build_prompt_session_context(
    session: Session,
    request: PromptSessionRequest,
) -> PromptSessionContext:
    """Create PromptSessionContext objects from an API request."""
    notification_config = (
        NotificationConfig(**request.notification_config)
        if request.notification_config
        else None
    )

    return PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        actor_agent_ids=request.actor_agent_ids,
        message=request.message,
        update_config=request.update_config,
        llm_config=request.llm_config,
        notification_config=notification_config,
        thinking_override=request.thinking,
        acting_user_id=request.acting_user_id or request.user_id,
        api_key_id=request.api_key_id,
        trigger=ObjectId(request.trigger) if request.trigger else None,
    )


def prepare_prompt_session(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
) -> Tuple[Session, PromptSessionContext]:
    """Initialize (or create) a session and build the runtime context."""
    session = setup_session(
        background_tasks=background_tasks,
        session_id=request.session_id,
        user_id=request.user_id,
        request=request,
    )
    context = build_prompt_session_context(session, request)
    return session, context


async def run_prompt_session(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    session_id = str(context.session.id) if context.session else None
    debugger = SessionDebugger(session_id)

    debugger.start_section("run_prompt_session")
    debugger.log("Non-streaming mode", emoji="info")

    async for data in _run_prompt_session_internal(
        context, background_tasks, stream=False
    ):
        # Pass session_id for SSE broadcasting
        update_type = data.get("type", "unknown")
        debugger.log(f"Emitting update: {update_type}", emoji="update")
        await emit_update(context.update_config, data, session_id=session_id)

    debugger.end_section("run_prompt_session")


async def run_prompt_session_stream(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    session_id = str(context.session.id) if context.session else None
    debugger = SessionDebugger(session_id)

    debugger.start_section("run_prompt_session_stream")
    try:
        async for data in _run_prompt_session_internal(
            context, background_tasks, stream=True
        ):
            # Also broadcast to SSE connections
            if session_id:
                try:
                    from eve.api.sse_manager import sse_manager

                    connection_count = sse_manager.get_connection_count(session_id)
                    debugger.log_sse_broadcast(session_id, data, connection_count)
                    await sse_manager.broadcast(session_id, data)
                except Exception as sse_error:
                    logger.error(f"Failed to broadcast to SSE: {sse_error}")
            yield data
    except Exception as e:
        traceback.print_exc()
        error_data = {
            "type": UpdateType.ERROR.value,
            "error": str(e),
            "update_config": context.update_config.model_dump()
            if context.update_config
            else None,
        }
        # Broadcast error to SSE as well
        session_id = str(context.session.id) if context.session else None
        if session_id:
            try:
                from eve.api.sse_manager import sse_manager

                await sse_manager.broadcast(session_id, error_data)
            except Exception:
                pass
        yield error_data
