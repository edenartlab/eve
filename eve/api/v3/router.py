import uuid
from typing import AsyncIterator

import sentry_sdk
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse

from eve import auth
from eve.agent.session.models import ChatMessageRequestInput
from eve.agent.session.orchestrator import (
    OrchestrationMode,
    OrchestrationRequest,
    orchestrate,
)
from eve.agent.session.setup import setup_session
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.errors import APIError, handle_errors
from eve.api.v3.models import SessionPromptRequest, SessionPromptResponse
from eve.async_mongo import get_async_collection
from eve.utils import dumps_json

router = APIRouter(prefix="/v3")


def _start_sentry_transaction(request: Request, name: str, op: str):
    if sentry_sdk is None:
        return None

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in {"sentry-trace", "baggage"}
    }

    continue_trace = getattr(sentry_sdk, "continue_trace", None)
    if callable(continue_trace):
        try:
            ctx = continue_trace(headers, name=name, op=op)
            transaction = sentry_sdk.start_transaction(ctx)
        except Exception:
            transaction = sentry_sdk.start_transaction(name=name, op=op)
    else:
        transaction = sentry_sdk.start_transaction(name=name, op=op)

    sentry_sdk.Hub.current.scope.span = transaction
    return transaction


async def _validate_session_exists(session_id: str) -> None:
    sessions = get_async_collection("sessions")
    try:
        session_oid = ObjectId(session_id)
    except Exception:
        raise APIError("Invalid session_id format", status_code=400)

    doc = await sessions.find_one({"_id": session_oid}, {"_id": 1})
    if not doc:
        raise APIError("Session not found", status_code=404)


def _build_prompt_request(request: SessionPromptRequest) -> PromptSessionRequest:
    if not request.user_id:
        raise APIError("user_id is required", status_code=400)

    message = ChatMessageRequestInput(
        content=request.message.content,
        attachments=request.message.attachments or [],
    )

    creation_args = (
        SessionCreationArgs(agent_ids=request.creation.agent_ids)
        if request.creation
        else None
    )

    return PromptSessionRequest(
        session_id=request.session_id,
        session_run_id=request.session_run_id,
        message=message,
        user_id=request.user_id,
        actor_agent_ids=request.actor_agent_ids,
        thinking=request.thinking,
        api_key_id=request.api_key_id,
        creation_args=creation_args,
    )


@router.post("/sessions/prompt", response_model=SessionPromptResponse)
@handle_errors
async def prompt_session_v3(
    request: SessionPromptRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    transaction = _start_sentry_transaction(
        raw_request, name="sessions.v3.prompt", op="http.server"
    )

    try:
        if not request.session_id and not request.creation:
            raise APIError(
                "creation.agent_ids required when session_id is missing",
                status_code=400,
            )

        if request.session_id:
            with sentry_sdk.start_span(op="v3.session.validate"):
                await _validate_session_exists(request.session_id)

        prompt_request = _build_prompt_request(request)
        session_run_id = request.session_run_id or str(uuid.uuid4())

        session = setup_session(
            background_tasks=background_tasks,
            session_id=prompt_request.session_id,
            user_id=prompt_request.user_id,
            request=prompt_request,
        )
        session_id = str(session.id)

        orch_request = OrchestrationRequest(
            initiating_user_id=prompt_request.user_id,
            session=session,
            actor_agent_ids=prompt_request.actor_agent_ids,
            message=prompt_request.message,
            llm_config=prompt_request.llm_config,
            update_config=prompt_request.update_config,
            notification_config=prompt_request.notification_config,
            thinking_override=prompt_request.thinking,
            acting_user_id=prompt_request.acting_user_id,
            api_key_id=prompt_request.api_key_id,
            trigger_id=prompt_request.trigger,
            mode=OrchestrationMode.API_REQUEST,
            stream=False,
            session_run_id=session_run_id,
            background_tasks=background_tasks,
        )

        async def run_orchestration():
            async for _ in orchestrate(orch_request):
                pass

        background_tasks.add_task(run_orchestration)

        return SessionPromptResponse(
            session_id=session_id, session_run_id=session_run_id
        )
    finally:
        if transaction:
            transaction.finish()


@router.post("/sessions/prompt/stream")
@handle_errors
async def prompt_session_v3_stream(
    request: SessionPromptRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request,
    _: dict = Depends(auth.authenticate_admin),
):
    transaction = _start_sentry_transaction(
        raw_request, name="sessions.v3.prompt.stream", op="http.server"
    )

    if request.session_id:
        with sentry_sdk.start_span(op="v3.session.validate"):
            await _validate_session_exists(request.session_id)

    if not request.session_id and not request.creation:
        raise APIError(
            "creation.agent_ids required when session_id is missing",
            status_code=400,
        )

    prompt_request = _build_prompt_request(request)
    session_run_id = request.session_run_id or str(uuid.uuid4())

    session = setup_session(
        background_tasks=background_tasks,
        session_id=prompt_request.session_id,
        user_id=prompt_request.user_id,
        request=prompt_request,
    )
    session_id = str(session.id)

    orch_request = OrchestrationRequest(
        initiating_user_id=prompt_request.user_id,
        session=session,
        actor_agent_ids=prompt_request.actor_agent_ids,
        message=prompt_request.message,
        llm_config=prompt_request.llm_config,
        update_config=prompt_request.update_config,
        notification_config=prompt_request.notification_config,
        thinking_override=prompt_request.thinking,
        acting_user_id=prompt_request.acting_user_id,
        api_key_id=prompt_request.api_key_id,
        trigger_id=prompt_request.trigger,
        mode=OrchestrationMode.API_REQUEST,
        stream=True,
        session_run_id=session_run_id,
        background_tasks=background_tasks,
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield f"data: {dumps_json({'event': 'session_created', 'data': {'session_id': session_id, 'session_run_id': session_run_id}})}\n\n"

            async for update in orchestrate(orch_request):
                yield f"data: {dumps_json({'event': 'update', 'data': update})}\n\n"

            yield f"data: {dumps_json({'event': 'done', 'data': ''})}\n\n"
        except Exception as exc:
            sentry_sdk.capture_exception(exc)
            yield f"data: {dumps_json({'event': 'error', 'data': {'error': str(exc)}})}\n\n"
        finally:
            if transaction:
                transaction.finish()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
