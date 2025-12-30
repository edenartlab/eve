"""
Unified LLM Orchestration Entry Point

This module provides a single, fully-instrumented entry point for all
LLM orchestration in Eve. All other entry points should call this function.

Design Principles:
1. Maximal observability by default (Sentry + Langfuse + structured logging)
2. Consistent interface across all use cases
3. Backward compatible with existing code
4. Clear separation of concerns (setup, execution, cleanup)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from bson import ObjectId
from fastapi import BackgroundTasks
from loguru import logger

from eve.agent import Agent
from eve.agent.session.context import add_chat_message
from eve.agent.session.instrumentation import PromptSessionInstrumentation
from eve.agent.session.models import (
    ChatMessageRequestInput,
    LLMConfig,
    NotificationConfig,
    PromptSessionContext,
    Session,
    SessionUpdateConfig,
)
from eve.agent.session.runtime import _run_prompt_session_internal
from eve.agent.session.setup import setup_session
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.tool import Tool


class OrchestrationMode(Enum):
    """Defines how orchestration was initiated."""

    API_REQUEST = "api_request"  # Standard API handler
    TRIGGER = "trigger"  # Scheduled trigger execution
    REMOTE_PROMPT = "remote_prompt"  # Modal remote invocation
    AUTOMATIC = "automatic"  # Automatic session step
    DEPLOYMENT = "deployment"  # Gateway (Discord/Telegram/etc.)
    AGENT_SESSION = "agent_session"  # Private agent workspace
    TOOL_CALLBACK = "tool_callback"  # Tool invoking another session


@dataclass
class OrchestrationRequest:
    """
    Unified request structure for all LLM orchestration.

    This replaces the various ad-hoc parameter passing patterns
    with a single, well-documented structure.
    """

    # Required: Who initiated this
    initiating_user_id: str

    # Session identification (one of these required)
    session_id: Optional[str] = None
    session: Optional[Session] = None  # Pre-loaded session

    # Session creation (if session_id is None and session is None)
    creation_args: Optional[Dict[str, Any]] = None

    # Agent specification
    actor_agent_ids: Optional[List[str]] = None  # Explicit actors
    agent: Optional[Agent] = None  # Pre-loaded agent (single actor)

    # Message to add
    message: Optional[ChatMessageRequestInput] = None

    # Configuration
    llm_config: Optional[LLMConfig] = None
    update_config: Optional[SessionUpdateConfig] = None
    notification_config: Optional[NotificationConfig] = None

    # Extra tools (e.g., for triggers with posting instructions)
    extra_tools: Optional[Dict[str, Tool]] = None

    # Execution mode
    mode: OrchestrationMode = OrchestrationMode.API_REQUEST
    stream: bool = False

    # Trigger-specific
    trigger_id: Optional[str] = None
    trigger_context: Optional[str] = None

    # Context window configuration
    selection_limit: Optional[int] = None  # Override default message selection limit

    # Thinking/reasoning
    thinking_override: Optional[bool] = None

    # Billing/auth
    acting_user_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Tracing (auto-generated if not provided)
    session_run_id: Optional[str] = None
    trace_name: Optional[str] = None

    # Background tasks (for memory formation, notifications)
    background_tasks: Optional[BackgroundTasks] = None

    # Instrumentation (auto-created if not provided)
    instrumentation: Optional[PromptSessionInstrumentation] = None


@dataclass
class OrchestrationResult:
    """Result from orchestration execution."""

    session_id: str
    session_run_id: str
    success: bool
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


async def orchestrate(
    request: OrchestrationRequest,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Single source of truth for LLM orchestration.

    This function:
    1. Sets up full instrumentation (Sentry, Langfuse, logging)
    2. Creates/loads session
    3. Adds user message (if provided)
    4. Runs the prompt loop
    5. Handles cleanup and error reporting

    Yields:
        SessionUpdate dictionaries

    Usage:
        # Streaming
        async for update in orchestrate(request):
            yield update

        # Non-streaming
        async for update in orchestrate(request):
            pass  # Updates are emitted via update_config
    """
    # === DEBUG: Entry point logging ===
    logger.info("[ORCHESTRATE] ========== START ==========")
    logger.info(f"[ORCHESTRATE] Mode: {request.mode.value}")
    logger.info(f"[ORCHESTRATE] User: {request.initiating_user_id}")
    logger.info(f"[ORCHESTRATE] Session ID (request): {request.session_id}")
    logger.info(
        f"[ORCHESTRATE] Session (pre-loaded): {request.session.id if request.session else None}"
    )
    logger.info(
        f"[ORCHESTRATE] Agent (pre-loaded): {request.agent.id if request.agent else None}"
    )
    logger.info(f"[ORCHESTRATE] Actor agent IDs: {request.actor_agent_ids}")
    logger.info(f"[ORCHESTRATE] Has message: {request.message is not None}")
    logger.info(
        f"[ORCHESTRATE] Message content preview: {request.message.content[:100] if request.message and request.message.content else 'N/A'}..."
    )
    logger.info(f"[ORCHESTRATE] Trigger ID: {request.trigger_id}")
    logger.info(f"[ORCHESTRATE] Stream: {request.stream}")
    logger.info(
        f"[ORCHESTRATE] Extra tools: {list(request.extra_tools.keys()) if request.extra_tools else None}"
    )

    # 1. Generate session_run_id if not provided
    session_run_id = request.session_run_id or str(uuid.uuid4())
    logger.info(f"[ORCHESTRATE] Session run ID: {session_run_id}")

    # 2. Resolve agent_id for instrumentation
    agent_id = None
    if request.agent:
        agent_id = str(request.agent.id)
    elif request.actor_agent_ids:
        agent_id = request.actor_agent_ids[0]
    logger.info(f"[ORCHESTRATE] Resolved agent ID for instrumentation: {agent_id}")

    # 3. Create or use provided instrumentation
    instrumentation = request.instrumentation
    if not instrumentation:
        logger.info("[ORCHESTRATE] Creating new instrumentation...")
        instrumentation = PromptSessionInstrumentation(
            session_id=request.session_id
            or (str(request.session.id) if request.session else None),
            session_run_id=session_run_id,
            user_id=request.initiating_user_id,
            agent_id=agent_id,
            trace_name=request.trace_name or f"orchestrate_{request.mode.value}",
            metadata={
                "mode": request.mode.value,
                "trigger_id": request.trigger_id,
                "has_message": request.message is not None,
            },
        )
        logger.info(
            f"[ORCHESTRATE] Instrumentation created: trace_name={instrumentation.trace_name}"
        )
    else:
        logger.info("[ORCHESTRATE] Using provided instrumentation")

    # 4. Ensure Sentry transaction exists
    logger.info(
        f"[ORCHESTRATE] Setting up Sentry transaction: orchestrate_{request.mode.value}"
    )
    instrumentation.ensure_sentry_transaction(
        name=f"orchestrate_{request.mode.value}",
        op="session.orchestrate",
    )
    instrumentation.add_breadcrumb(
        f"Orchestration started: mode={request.mode.value}",
        {
            "session_id": request.session_id,
            "trigger_id": request.trigger_id,
        },
    )
    logger.info("[ORCHESTRATE] Sentry transaction started")

    success = True
    session = None
    update_count = 0
    background_tasks = request.background_tasks or BackgroundTasks()

    try:
        # 5. Setup/load session
        logger.info("[ORCHESTRATE] Stage: setup_session - START")
        setup_stage = instrumentation.track_stage("setup_session", level="info")
        with setup_stage:
            if request.session:
                session = request.session
                logger.info(
                    f"[ORCHESTRATE] Using provided session: id={session.id}, title={session.title}"
                )
            else:
                # Build a PromptSessionRequest for setup_session
                logger.info(
                    "[ORCHESTRATE] Building PromptSessionRequest for setup_session..."
                )
                prompt_request = _to_prompt_session_request(request)
                session = setup_session(
                    background_tasks=background_tasks,
                    session_id=request.session_id,
                    user_id=request.initiating_user_id,
                    request=prompt_request,
                )
                logger.info(
                    f"[ORCHESTRATE] Created/loaded session: id={session.id}, title={session.title}"
                )

            instrumentation.update_context(session_id=str(session.id))
        logger.info("[ORCHESTRATE] Stage: setup_session - END")

        # 6. Build context
        logger.info("[ORCHESTRATE] Stage: build_context - START")
        context_stage = instrumentation.track_stage("build_context", level="info")
        with context_stage:
            context = _build_prompt_context(session, request, session_run_id)
            context.instrumentation = instrumentation
            logger.info(
                f"[ORCHESTRATE] Context built: actor_agent_ids={context.actor_agent_ids}"
            )
        logger.info("[ORCHESTRATE] Stage: build_context - END")

        # 7. Add message if provided
        user_message_id = None
        if request.message and request.initiating_user_id:
            logger.info("[ORCHESTRATE] Stage: add_message - START")
            message_stage = instrumentation.track_stage("add_message", level="info")
            with message_stage:
                user_message = await add_chat_message(session, context)
                user_message_id = str(user_message.id) if user_message else None
                logger.info(
                    f"[ORCHESTRATE] Added user message to session {session.id}, message_id={user_message_id}"
                )
            logger.info("[ORCHESTRATE] Stage: add_message - END")

            # For trigger mode, yield the user message ID so it can be recorded
            if request.mode == OrchestrationMode.TRIGGER and user_message_id:
                yield {
                    "type": "trigger_message_created",
                    "message_id": user_message_id,
                }
        else:
            logger.info("[ORCHESTRATE] Skipping add_message (no message or no user)")

        instrumentation.add_breadcrumb(
            "Starting prompt session runtime",
            {"session_id": str(session.id)},
        )

        # 8. Run orchestration
        logger.info("[ORCHESTRATE] Stage: runtime_execution - START")
        logger.info("[ORCHESTRATE] Calling _run_prompt_session_internal...")
        async for update in _run_prompt_session_internal(
            context,
            background_tasks,
            stream=request.stream,
            instrumentation=instrumentation,
        ):
            update_count += 1
            update_type = update.get("type", "unknown")
            logger.info(
                f"[ORCHESTRATE] Yielded update #{update_count}: type={update_type}"
            )
            yield update

        logger.info(
            f"[ORCHESTRATE] Stage: runtime_execution - END (yielded {update_count} updates)"
        )

    except Exception as e:
        success = False
        logger.error(f"[ORCHESTRATE] !!!!! ERROR !!!!! {type(e).__name__}: {e}")
        logger.error(f"[ORCHESTRATE] Error occurred after {update_count} updates")
        instrumentation.log_error("Orchestration failed", e)
        raise

    finally:
        instrumentation.add_breadcrumb(
            f"Orchestration completed: success={success}",
            {"session_id": str(session.id) if session else None},
        )
        instrumentation.finalize(success=success)
        logger.info("[ORCHESTRATE] ========== END ==========")
        logger.info(f"[ORCHESTRATE] Session: {session.id if session else None}")
        logger.info(f"[ORCHESTRATE] Success: {success}")
        logger.info(f"[ORCHESTRATE] Total updates yielded: {update_count}")


# --- Convenience Wrappers ---


async def orchestrate_trigger(
    trigger_id: str,
    trigger_prompt: Optional[str],
    session: Session,
    agent: Agent,
    user_id: str,
    *,
    extra_tools: Optional[Dict[str, Tool]] = None,
    trigger_context: Optional[str] = None,
    notification_config: Optional[NotificationConfig] = None,
    update_config: Optional[SessionUpdateConfig] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience wrapper for trigger execution.

    If trigger_prompt is None, skips message creation (message already added by caller).
    """
    logger.info(f"[ORCHESTRATE_TRIGGER] Called for trigger_id={trigger_id}")
    logger.info(
        f"[ORCHESTRATE_TRIGGER] Session: {session.id}, Agent: {agent.id} ({agent.username})"
    )

    request = OrchestrationRequest(
        initiating_user_id=user_id,
        session=session,
        agent=agent,
        actor_agent_ids=[str(agent.id)],
        message=ChatMessageRequestInput(role="eden", content=trigger_prompt)
        if trigger_prompt
        else None,
        mode=OrchestrationMode.TRIGGER,
        trigger_id=trigger_id,
        trigger_context=trigger_context,
        extra_tools=extra_tools,
        notification_config=notification_config,
        update_config=update_config,
        background_tasks=background_tasks,
        trace_name=f"trigger_{trigger_id[:8]}",
    )

    update_count = 0
    async for update in orchestrate(request):
        update_count += 1
        yield update

    logger.info(
        f"[ORCHESTRATE_TRIGGER] Completed for trigger_id={trigger_id}, updates={update_count}"
    )


async def orchestrate_remote(
    session_id: str,
    agent_id: str,
    user_id: str,
    content: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    extra_tools: Optional[List[str]] = None,
    selection_limit: Optional[int] = None,
) -> None:
    """Convenience wrapper for remote prompt sessions (Modal invocation)."""
    logger.info(f"[ORCHESTRATE_REMOTE] Called for session_id={session_id}")
    logger.info(f"[ORCHESTRATE_REMOTE] Agent: {agent_id}, User: {user_id}")
    logger.info(
        f"[ORCHESTRATE_REMOTE] Content: {content[:100] if content else 'None'}..."
    )
    logger.info(
        f"[ORCHESTRATE_REMOTE] Attachments: {len(attachments) if attachments else 0}"
    )

    message = None
    if content or attachments:
        message = ChatMessageRequestInput(
            role="user",
            content=content or "",
            attachments=attachments or [],
        )

    extra_tools_dict = None
    if extra_tools:
        extra_tools_dict = {k: Tool.load(k) for k in extra_tools}
        logger.info(f"[ORCHESTRATE_REMOTE] Extra tools loaded: {extra_tools}")

    request = OrchestrationRequest(
        initiating_user_id=user_id,
        session_id=session_id,
        actor_agent_ids=[agent_id],
        message=message,
        extra_tools=extra_tools_dict,
        mode=OrchestrationMode.REMOTE_PROMPT,
        trace_name=f"remote_{session_id[:8]}",
        selection_limit=selection_limit,
    )

    update_count = 0
    async for _ in orchestrate(request):
        update_count += 1

    logger.info(
        f"[ORCHESTRATE_REMOTE] Completed for session_id={session_id}, updates={update_count}"
    )


async def orchestrate_automatic(
    session: Session,
    actor: Agent,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience wrapper for automatic session steps."""
    logger.info(f"[ORCHESTRATE_AUTOMATIC] Called for session={session.id}")
    logger.info(f"[ORCHESTRATE_AUTOMATIC] Actor: {actor.id} ({actor.username})")
    logger.info(f"[ORCHESTRATE_AUTOMATIC] Session owner: {session.owner}")

    request = OrchestrationRequest(
        initiating_user_id=str(session.owner),
        session=session,
        agent=actor,
        actor_agent_ids=[str(actor.id)],
        message=ChatMessageRequestInput(role="system", content=""),
        mode=OrchestrationMode.AUTOMATIC,
        background_tasks=BackgroundTasks(),
        trace_name=f"automatic_{str(session.id)[:8]}",
    )

    update_count = 0
    async for update in orchestrate(request):
        update_count += 1
        yield update

    logger.info(
        f"[ORCHESTRATE_AUTOMATIC] Completed for session={session.id}, updates={update_count}"
    )


async def orchestrate_deployment(
    session: Session,
    agent: Agent,
    user_id: str,
    message: Optional[ChatMessageRequestInput] = None,
    *,
    update_config: Optional[SessionUpdateConfig] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience wrapper for deployment-originated requests (Discord, Telegram, etc.).

    Args:
        message: Optional. If provided, will be added to session. If None, assumes
                 message was already added to session (e.g., for Twitter where we need
                 the message ID before orchestration).
    """
    logger.info(f"[ORCHESTRATE_DEPLOYMENT] Called for session={session.id}")
    logger.info(f"[ORCHESTRATE_DEPLOYMENT] Agent: {agent.id} ({agent.username})")
    logger.info(f"[ORCHESTRATE_DEPLOYMENT] User: {user_id}")
    logger.info(
        f"[ORCHESTRATE_DEPLOYMENT] Message: {message.content[:100] if message and message.content else 'None (pre-added)'}..."
    )

    request = OrchestrationRequest(
        initiating_user_id=user_id,
        session=session,
        agent=agent,
        actor_agent_ids=[str(agent.id)],
        message=message,
        mode=OrchestrationMode.DEPLOYMENT,
        update_config=update_config,
        background_tasks=background_tasks,
        trace_name=f"deployment_{str(session.id)[:8]}",
    )

    update_count = 0
    async for update in orchestrate(request):
        update_count += 1
        yield update

    logger.info(
        f"[ORCHESTRATE_DEPLOYMENT] Completed for session={session.id}, updates={update_count}"
    )


async def orchestrate_agent_session(
    agent_session: Session,
    actor: Agent,
    parent_session: Session,
    context: PromptSessionContext,
    *,
    stream: bool = False,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience wrapper for agent private workspace execution.

    This is used when an agent runs in their private workspace (agent_session)
    as part of a multi-agent conversation in the parent session.
    """
    logger.info(
        f"[ORCHESTRATE_AGENT_SESSION] Called for agent_session={agent_session.id}"
    )
    logger.info(f"[ORCHESTRATE_AGENT_SESSION] Actor: {actor.id} ({actor.username})")
    logger.info(f"[ORCHESTRATE_AGENT_SESSION] Parent session: {parent_session.id}")
    logger.info(f"[ORCHESTRATE_AGENT_SESSION] Stream: {stream}")

    request = OrchestrationRequest(
        initiating_user_id=str(parent_session.owner),
        session=agent_session,
        agent=actor,
        actor_agent_ids=[str(actor.id)],
        message=context.message,
        mode=OrchestrationMode.AGENT_SESSION,
        stream=stream,
        extra_tools=context.extra_tools,
        update_config=context.update_config,
        llm_config=context.llm_config,
        background_tasks=BackgroundTasks(),
        trace_name=f"agent_session_{str(agent_session.id)[:8]}",
    )

    update_count = 0
    async for update in orchestrate(request):
        update_count += 1
        yield update

    logger.info(
        f"[ORCHESTRATE_AGENT_SESSION] Completed for agent_session={agent_session.id}, updates={update_count}"
    )


# --- Helper Functions ---


def _to_prompt_session_request(request: OrchestrationRequest) -> PromptSessionRequest:
    """Convert OrchestrationRequest to PromptSessionRequest for setup_session."""
    creation_args = None
    if request.creation_args:
        creation_args = SessionCreationArgs(**request.creation_args)

    return PromptSessionRequest(
        user_id=request.initiating_user_id,
        session_id=request.session_id,
        creation_args=creation_args,
        actor_agent_ids=request.actor_agent_ids,
        message=request.message,
        llm_config=request.llm_config,
        update_config=request.update_config,
        notification_config=request.notification_config,
        thinking=request.thinking_override,
        acting_user_id=request.acting_user_id,
        api_key_id=request.api_key_id,
        trigger=request.trigger_id,
    )


def _build_prompt_context(
    session: Session,
    request: OrchestrationRequest,
    session_run_id: str,
) -> PromptSessionContext:
    """Build PromptSessionContext from OrchestrationRequest."""
    return PromptSessionContext(
        session=session,
        initiating_user_id=request.initiating_user_id,
        actor_agent_ids=request.actor_agent_ids,
        message=request.message,
        update_config=request.update_config,
        llm_config=request.llm_config,
        notification_config=request.notification_config,
        thinking_override=request.thinking_override,
        extra_tools=request.extra_tools,
        acting_user_id=request.acting_user_id or request.initiating_user_id,
        api_key_id=request.api_key_id,
        trigger=ObjectId(request.trigger_id) if request.trigger_id else None,
        session_run_id=session_run_id,
        selection_limit=request.selection_limit,
    )
