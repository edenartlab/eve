from dataclasses import dataclass
from typing import AsyncIterator, Optional, Tuple

from bson import ObjectId
from contextlib import nullcontext
from fastapi import BackgroundTasks

from eve.agent.session.models import (
    NotificationConfig,
    PromptSessionContext,
    Session,
    UpdateType,
)
from eve.agent.session_new.instrumentation import PromptSessionInstrumentation
from eve.agent.session_new.runtime import _run_prompt_session_internal
from eve.api.api_requests import PromptSessionRequest
from eve.api.helpers import emit_update
from loguru import logger
import uuid
from eve.agent.session_new.setup import setup_session


@dataclass
class PromptSessionHandle:
    """Unified entrypoint for preparing and executing a prompt session."""

    session: Session
    context: PromptSessionContext
    background_tasks: BackgroundTasks
    instrumentation: Optional[PromptSessionInstrumentation] = None

    @property
    def session_id(self) -> Optional[str]:
        return str(self.session.id) if self.session else None

    def iter_updates(self, stream: bool = False) -> AsyncIterator[dict]:
        """Expose the raw session update generator for advanced callers."""
        return _run_prompt_session_internal(
            self.context,
            self.background_tasks,
            stream=stream,
            instrumentation=self.instrumentation,
        )

    async def run(self) -> None:
        """Default non-streaming execution that emits updates to configured channels."""
        session_id = self.session_id
        success = True
        inst = self.instrumentation
        stage_cm = inst.track_stage("handle.run", level="info") if inst else nullcontext()
        try:
            with stage_cm:
                async for data in self.iter_updates(stream=False):
                    await emit_update(self.context.update_config, data, session_id=session_id)
        except Exception:
            success = False
            raise
        finally:
            if inst:
                inst.finalize(success=success)

    async def stream_updates(self) -> AsyncIterator[dict]:
        """Streaming execution that also mirrors updates to SSE subscribers."""
        session_id = self.session_id
        inst = self.instrumentation
        success = True
        stage_cm = inst.track_stage("handle.stream_updates", level="info") if inst else nullcontext()
        try:
            with stage_cm:
                async for data in self.iter_updates(stream=True):
                    if session_id:
                        try:
                            from eve.api.sse_manager import sse_manager

                            await sse_manager.broadcast(session_id, data)
                        except Exception as sse_error:
                            logger.error(f"Failed to broadcast to SSE: {sse_error}")
                    yield data
        except Exception as e:
            success = False
            logger.exception("Error during prompt session stream")
            error_data = {
                "type": UpdateType.ERROR.value,
                "error": str(e),
                "update_config": self.context.update_config.model_dump()
                if self.context.update_config
                else None,
            }
            if session_id:
                try:
                    from eve.api.sse_manager import sse_manager

                    await sse_manager.broadcast(session_id, error_data)
                except Exception:
                    logger.error(f"Failed to broadcast error to SSE: {e}")
            yield error_data
        finally:
            if inst:
                inst.finalize(success=success)


def create_prompt_session_handle(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
) -> PromptSessionHandle:
    """Create a prepared session handle that callers can execute or stream."""
    instrumentation = PromptSessionInstrumentation(
        session_id=request.session_id,
        session_run_id=None,
        user_id=request.user_id,
        agent_id=request.actor_agent_ids[0] if request.actor_agent_ids else None,
    )
    setup_stage = instrumentation.track_stage("setup_session", level="info")
    with setup_stage:
        session = setup_session(
            background_tasks=background_tasks,
            session_id=request.session_id,
            user_id=request.user_id,
            request=request,
        )
    context_stage = instrumentation.track_stage("build_prompt_session_context", level="info")
    with context_stage:
        context = build_prompt_session_context(session, request)
    instrumentation.update_context(
        session_id=str(session.id) if session else None,
        session_run_id=context.session_run_id,
        user_id=context.initiating_user_id,
    )
    context.instrumentation = instrumentation
    return PromptSessionHandle(
        session=session,
        context=context,
        background_tasks=background_tasks,
        instrumentation=instrumentation,
    )


def build_prompt_session_context(
    session: Session,
    request: PromptSessionRequest,
) -> PromptSessionContext:
    """Create PromptSessionContext objects from an API request."""
    session_run_id = str(uuid.uuid4())
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
        session_run_id=session_run_id,
    )


def prepare_prompt_session(
    request: PromptSessionRequest,
    background_tasks: BackgroundTasks,
) -> Tuple[Session, PromptSessionContext]:
    """Initialize (or create) a session and build the runtime context."""
    handle = create_prompt_session_handle(request, background_tasks)
    return handle.session, handle.context


async def run_prompt_session(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    instrumentation = getattr(context, "instrumentation", None)
    handle = PromptSessionHandle(
        context.session, context, background_tasks, instrumentation=instrumentation
    )
    await handle.run()


async def run_prompt_session_stream(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    instrumentation = getattr(context, "instrumentation", None)
    handle = PromptSessionHandle(
        context.session, context, background_tasks, instrumentation=instrumentation
    )
    async for data in handle.stream_updates():
        yield data
