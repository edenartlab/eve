import asyncio
import json
import os
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import BackgroundTasks
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.llm.llm import async_prompt as provider_async_prompt
from eve.agent.llm.llm import async_prompt_stream as provider_async_prompt_stream
from eve.agent.llm.llm import get_provider
from eve.agent.memory.memory_models import select_messages
from eve.agent.memory.service import memory_service
from eve.agent.session.debug_logger import SessionDebugger
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageObservability,
    LLMContext,
    LLMUsage,
    PromptSessionContext,
    Session,
    SessionMemoryContext,
    SessionUpdate,
    ToolCall,
    UpdateType,
)
from eve.agent.session.tracing import trace_async_operation
from eve.utils import dumps_json

from .budget import update_session_budget
from .context import (
    build_llm_context,
    convert_message_roles,
    determine_actors,
    label_message_channels,
)
from .instrumentation import PromptSessionInstrumentation
from .notifications import (
    _send_session_notification,
    check_if_session_active,
    create_session_message_notification,
)
from .tools import process_tool_calls
from .util import validate_prompt_session


class SessionCancelledException(Exception):
    """Exception raised when a session is cancelled via Ably signal."""


class PromptSessionRuntime:
    """Encapsulates the single-actor prompt session loop."""

    def __init__(
        self,
        session: Session,
        llm_context: LLMContext,
        actor: Agent,
        *,
        stream: bool,
        is_client_platform: bool,
        session_run_id: Optional[str],
        api_key_id: Optional[str],
        instrumentation: Optional[PromptSessionInstrumentation] = None,
    ):
        self.session = session
        self.llm_context = llm_context
        self.actor = actor
        self.stream = stream
        self.is_client_platform = is_client_platform
        self.session_run_id = session_run_id or str(uuid.uuid4())
        self.api_key_id = api_key_id
        self.instrumentation = instrumentation
        self.debugger = (
            instrumentation.debugger if instrumentation else SessionDebugger()
        )
        self.cancellation_event = asyncio.Event()
        self.tool_cancellation_events: Dict[str, asyncio.Event] = {}
        self.tool_was_cancelled = False
        self.ably_client = None
        self.transaction = None
        self.active_request_registered = False
        self._last_stream_result: Optional[Dict[str, Any]] = None

    async def run(self):
        """Async generator that yields SessionUpdates."""
        try:
            self._start_transaction()
            await self._setup_cancellation_listener()
            async for update in self._prompt_loop():
                yield update
        except SessionCancelledException:
            async for update in self._handle_session_cancelled():
                yield update
        finally:
            await self._cleanup()

    async def _prompt_loop(self):
        stage_cm = (
            self.instrumentation.track_stage("runtime.loop", level="debug")
            if self.instrumentation
            else nullcontext()
        )
        with stage_cm:
            self._register_active_request()
            yield self._start_update()

            prompt_session_finished = False
            while not prompt_session_finished:
                self._ensure_not_cancelled()
                await self._refresh_llm_messages()
                self._maybe_disable_tools()

                provider = self._select_provider()
                llm_result: Dict[str, Any]

                if self.stream:
                    self._last_stream_result = None
                    async for update in self._stream_llm_response(provider):
                        yield update
                    llm_result = self._last_stream_result or {}
                else:
                    llm_result = await self._non_stream_llm_response(provider)

                assistant_message = await self._persist_assistant_message(llm_result)
                yield SessionUpdate(
                    type=UpdateType.ASSISTANT_MESSAGE,
                    message=assistant_message,
                    agent={
                        "_id": str(self.actor.id),
                        "username": self.actor.username,
                        "name": self.actor.name,
                        "userImage": self.actor.userImage,
                    },
                    session_run_id=self.session_run_id,
                )

                await self._maybe_notify_user()

                async for update in self._process_tool_calls(assistant_message):
                    yield update

                if llm_result.get("stop_reason") in ["stop", "completed"]:
                    prompt_session_finished = True

                self.llm_context.tool_choice = "auto"

            yield SessionUpdate(
                type=UpdateType.END_PROMPT, session_run_id=self.session_run_id
            )

    def _start_update(self) -> SessionUpdate:
        agent_payload = (
            {
                "_id": str(self.actor.id),
                "username": self.actor.username,
                "name": self.actor.name,
                "userImage": self.actor.userImage,
            }
            if self.actor
            else None
        )
        return SessionUpdate(
            type=UpdateType.START_PROMPT,
            agent=agent_payload,
            session_run_id=self.session_run_id,
        )

    async def _refresh_llm_messages(self):
        fresh_messages = select_messages(self.session)
        system_message = self.llm_context.messages[0]
        system_extras = []

        for msg in self.llm_context.messages[1:]:
            if msg.role == "system":
                system_extras.append(msg)
            else:
                break

        if self.session.trigger:
            from eve.trigger import Trigger

            trigger = Trigger.from_mongo(self.session.trigger)
            trigger_message = ChatMessage(
                session=self.session.id,
                role="system",
                content=f"<Full Task Context>\n{trigger.context}\n</Full Task Context>",
            )
            system_extras.append(trigger_message)

        refreshed_messages = [system_message]
        if system_extras:
            refreshed_messages.extend(system_extras)
        refreshed_messages.extend(fresh_messages)
        refreshed_messages = label_message_channels(refreshed_messages)
        refreshed_messages = convert_message_roles(refreshed_messages, self.actor.id)
        self.llm_context.messages = refreshed_messages
        self.llm_context.metadata.generation_id = str(uuid.uuid4())

    def _maybe_disable_tools(self):
        if self.tool_was_cancelled:
            self.llm_context.tools = {}
            self.llm_context.tool_choice = "none"

    def _select_provider(self):
        provider = get_provider(self.llm_context, instrumentation=self.instrumentation)
        provider_label = provider.__class__.__name__ if provider else "LegacyLiteLLM"
        self.debugger.log(
            "Selected LLM provider",
            {"provider": provider_label},
            emoji="llm",
        )
        return provider

    async def _stream_llm_response(self, provider):
        content = ""
        tool_calls_dict: Dict[int, Dict[str, Any]] = {}
        stop_reason = None
        tokens_spent = 0

        async with trace_async_operation(
            "llm.stream", model=self.llm_context.config.model
        ):
            stream_iter = provider_async_prompt_stream(self.llm_context, provider)
            async for chunk in stream_iter:
                self._ensure_not_cancelled()

                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        content += choice.delta.content
                        yield SessionUpdate(
                            type=UpdateType.ASSISTANT_TOKEN,
                            text=choice.delta.content,
                            session_run_id=self.session_run_id,
                        )
                    if choice.delta and choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            if tc.index not in tool_calls_dict:
                                tool_calls_dict[tc.index] = {
                                    "id": tc.id,
                                    "name": tc.function.name if tc.function else None,
                                    "arguments": "",
                                }
                            if tc.function and tc.function.arguments:
                                tool_calls_dict[tc.index]["arguments"] += (
                                    tc.function.arguments
                                )
                    if choice.finish_reason:
                        stop_reason = choice.finish_reason

                if hasattr(chunk, "usage") and chunk.usage:
                    tokens_spent = chunk.usage.total_tokens

        tool_calls = self._materialize_tool_calls(tool_calls_dict)
        usage_payload = LLMUsage(
            total_tokens=tokens_spent,
            prompt_tokens=None,
            completion_tokens=None,
        )
        self._last_stream_result = {
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
            "tokens_spent": tokens_spent,
            "usage": usage_payload.model_dump(),
        }
        return

    async def _non_stream_llm_response(self, provider) -> Dict[str, Any]:
        async with trace_async_operation(
            "llm.prompt", model=self.llm_context.config.model
        ):
            response = await provider_async_prompt(self.llm_context, provider)

        usage_dump = (
            response.usage.model_dump()
            if hasattr(response, "usage") and response.usage
            else None
        )

        return {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "stop_reason": response.stop,
            "tokens_spent": response.tokens_spent,
            "thought": response.thought,
            "usage": usage_dump,
        }

    def _materialize_tool_calls(
        self, tool_calls_dict: Dict[int, Dict[str, Any]]
    ) -> Optional[List[ToolCall]]:
        if not tool_calls_dict:
            return None

        tool_calls: List[ToolCall] = []
        for idx in sorted(tool_calls_dict.keys()):
            tc_data = tool_calls_dict[idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    tool=tc_data["name"],
                    args=args,
                    status="pending",
                )
            )
        return tool_calls

    async def _persist_assistant_message(
        self, llm_result: Dict[str, Any]
    ) -> ChatMessage:
        usage_payload = llm_result.get("usage")
        usage_obj = None
        if usage_payload:
            if isinstance(usage_payload, dict):
                usage_obj = LLMUsage(**usage_payload)
            elif isinstance(usage_payload, LLMUsage):
                usage_obj = usage_payload

        assistant_message = ChatMessage(
            session=self.session.id,
            sender=ObjectId(self.llm_context.metadata.trace_metadata.agent_id),
            role="assistant",
            content=llm_result["content"],
            tool_calls=llm_result.get("tool_calls"),
            finish_reason=llm_result.get("stop_reason"),
            thought=llm_result.get("thought"),
            llm_config=self.llm_context.config.__dict__
            if self.llm_context.config
            else None,
            observability=ChatMessageObservability(
                session_id=self.llm_context.metadata.session_id,
                trace_id=self.llm_context.metadata.trace_id,
                generation_id=self.llm_context.metadata.generation_id,
                session_run_id=self.session_run_id,
                tokens_spent=llm_result.get("tokens_spent"),
                prompt_tokens=(
                    usage_obj.prompt_tokens
                    if usage_obj
                    else llm_result.get("prompt_tokens")
                ),
                completion_tokens=(
                    usage_obj.completion_tokens
                    if usage_obj
                    else llm_result.get("completion_tokens")
                ),
                cached_prompt_tokens=(
                    usage_obj.cached_prompt_tokens if usage_obj else None
                ),
                cached_completion_tokens=(
                    usage_obj.cached_completion_tokens if usage_obj else None
                ),
                cost_usd=usage_obj.cost_usd if usage_obj else None,
                usage=usage_obj,
            ),
            apiKey=ObjectId(self.api_key_id) if self.api_key_id else None,
        )
        assistant_message.save()

        memory_context = self.session.memory_context
        memory_context.last_activity = datetime.now(timezone.utc)
        memory_context.messages_since_memory_formation += 1
        self.session.update(memory_context=memory_context.model_dump())
        self.session.memory_context = SessionMemoryContext(
            **self.session.memory_context
        )

        update_session_budget(
            self.session,
            tokens_spent=llm_result.get("tokens_spent", 0),
            turns_spent=1,
        )

        self._record_transaction_metadata(assistant_message, llm_result)
        return assistant_message

    def _record_transaction_metadata(
        self, assistant_message: ChatMessage, llm_result: Dict[str, Any]
    ):
        if not self.transaction:
            return

        tokens_spent = llm_result.get("tokens_spent")
        stop_reason = llm_result.get("stop_reason")
        if tokens_spent is not None:
            self.transaction.set_data("tokens_spent", tokens_spent)
        if stop_reason is not None:
            self.transaction.set_data("stop_reason", stop_reason)
        if assistant_message.tool_calls:
            self.transaction.set_data(
                "tool_calls_count", len(assistant_message.tool_calls)
            )

        if (
            assistant_message.observability
            and hasattr(self.transaction, "trace_id")
            and self.transaction.trace_id
        ):
            assistant_message.observability.sentry_trace_id = self.transaction.trace_id
            assistant_message.save()

    async def _maybe_notify_user(self):
        try:
            is_active_response = await check_if_session_active(
                str(self.session.owner), str(self.session.id)
            )
            is_active = is_active_response.get("is_active", False)
            if not is_active:
                await create_session_message_notification(
                    user_id=str(self.session.owner),
                    session_id=str(self.session.id),
                    agent_id=str(self.actor.id),
                )
        except Exception as e:
            logger.warning(
                f"[NOTIFICATION] ❌ Failed to create session message notification: {e}"
            )

    async def _process_tool_calls(self, assistant_message: ChatMessage):
        if not assistant_message.tool_calls:
            return

        async with trace_async_operation(
            "tools.process_all", tool_count=len(assistant_message.tool_calls)
        ):
            async for update in process_tool_calls(
                self.session,
                assistant_message,
                self.llm_context,
                self.cancellation_event,
                self.tool_cancellation_events,
                self.is_client_platform,
                self.session_run_id,
            ):
                self._ensure_not_cancelled()
                if update.type == UpdateType.TOOL_CANCELLED:
                    self.tool_was_cancelled = True
                yield update

    def _register_active_request(self):
        active_requests = self.session.active_requests or []
        active_requests.append(self.session_run_id)
        self.session.update(active_requests=active_requests)
        self.active_request_registered = True

    def _remove_active_request(self):
        if not self.active_request_registered:
            return
        active_requests = self.session.active_requests or []
        if self.session_run_id in active_requests:
            active_requests.remove(self.session_run_id)
            self.session.update(active_requests=active_requests)
        self.active_request_registered = False

    def _start_transaction(self):
        if self.instrumentation:
            transaction = self.instrumentation.ensure_sentry_transaction(
                name="prompt_session", op="session.prompt"
            )
            if transaction:
                transaction.set_tag("stream", str(self.stream))
            self.transaction = transaction
            return
        try:
            import sentry_sdk
        except ImportError:
            return

        transaction = sentry_sdk.start_transaction(
            name="prompt_session",
            op="session.prompt",
        )
        if transaction:
            transaction.set_tag("session_id", str(self.session.id))
            if self.llm_context.metadata and self.llm_context.metadata.trace_metadata:
                transaction.set_tag(
                    "user_id", str(self.llm_context.metadata.trace_metadata.user_id)
                )
            transaction.set_tag("agent_id", str(self.actor.id))
            transaction.set_tag("session_run_id", self.session_run_id)
            transaction.set_tag("stream", str(self.stream))
            sentry_sdk.Hub.current.scope.span = transaction
        self.transaction = transaction

    async def _setup_cancellation_listener(self):
        try:
            from ably import AblyRealtime

            self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
            channel_name = f"{os.getenv('DB')}-session-cancel-{self.session.id}"
            channel = self.ably_client.channels.get(channel_name)

            async def cancellation_handler(message):
                try:
                    data = message.data
                    if not (
                        isinstance(data, dict)
                        and data.get("session_id") == str(self.session.id)
                    ):
                        return
                    tool_call_id = data.get("tool_call_id")
                    if tool_call_id is not None:
                        if tool_call_id in self.tool_cancellation_events:
                            self.tool_cancellation_events[tool_call_id].set()
                        return

                    cancel_trace_id = data.get("trace_id")
                    if (
                        cancel_trace_id is None
                        or cancel_trace_id == self.session_run_id
                    ):
                        self.cancellation_event.set()
                except Exception as e:
                    logger.error(f"Error in cancellation handler: {e}")

            await channel.subscribe("cancel", cancellation_handler)
        except Exception as e:
            logger.error(
                f"Failed to setup Ably cancellation for session {self.session.id}: {e}"
            )

    async def _handle_session_cancelled(self):
        try:
            last_message = self._fetch_last_message()
            if last_message and last_message.tool_calls:
                updated = False
                for idx, tool_call in enumerate(last_message.tool_calls):
                    if tool_call.status in ["pending", "running"]:
                        tool_call.status = "cancelled"
                        updated = True
                        yield SessionUpdate(
                            type=UpdateType.TOOL_CANCELLED,
                            tool_name=tool_call.tool,
                            tool_index=idx,
                            result={"status": "cancelled"},
                            session_run_id=self.session_run_id,
                        )

                if updated:
                    try:
                        messages_collection = ChatMessage.get_collection()
                        messages_collection.update_one(
                            {"_id": last_message.id},
                            {
                                "$set": {
                                    "tool_calls": [
                                        tc.model_dump()
                                        for tc in last_message.tool_calls
                                    ]
                                }
                            },
                        )
                    except Exception:
                        last_message.save()

            cancel_message = ChatMessage(
                session=self.session.id,
                sender=ObjectId("000000000000000000000000"),
                role="system",
                content="Response cancelled by user",
            )
            cancel_message.save()

            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE,
                message=cancel_message,
                session_run_id=self.session_run_id,
            )
            yield SessionUpdate(
                type=UpdateType.END_PROMPT, session_run_id=self.session_run_id
            )
        except Exception as e:
            logger.error(f"Error during session cancellation cleanup: {e}")
            yield SessionUpdate(
                type=UpdateType.END_PROMPT, session_run_id=self.session_run_id
            )

    def _fetch_last_message(self) -> Optional[ChatMessage]:
        last_messages = ChatMessage.find(
            {"session": self.session.id},
            sort="createdAt",
            desc=True,
            limit=1,
        )
        return last_messages[0] if last_messages else None

    def _ensure_not_cancelled(self):
        if self.cancellation_event.is_set():
            raise SessionCancelledException("Session cancelled by user")

    async def _cleanup(self):
        self._remove_active_request()
        if self.ably_client:
            try:
                await self.ably_client.close()
            except Exception as e:
                logger.error(f"Error closing Ably client: {e}")
            finally:
                self.ably_client = None

        if self.transaction and not self.instrumentation:
            self.transaction.finish()
        self.transaction = None


async def async_prompt_session(
    session: Session,
    llm_context: LLMContext,
    agent: Agent,
    stream: bool = False,
    is_client_platform: bool = False,
    session_run_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    runtime = PromptSessionRuntime(
        session,
        llm_context,
        agent,
        stream=stream,
        is_client_platform=is_client_platform,
        session_run_id=session_run_id,
        api_key_id=api_key_id,
        instrumentation=instrumentation,
    )
    async for update in runtime.run():
        yield update


def format_session_update(update: SessionUpdate, context: PromptSessionContext) -> dict:
    """Convert SessionUpdate to the format expected by handlers"""
    data = {
        "type": update.type.value,
        "update_config": context.update_config.model_dump()
        if context.update_config
        else None,
    }

    # Include session_run_id in all updates for request tracking
    if update.session_run_id:
        data["session_run_id"] = update.session_run_id

    if update.type == UpdateType.START_PROMPT:
        if update.agent:
            data["agent"] = update.agent
        # Include session_id in start_prompt event for frontend to capture
        data["session_id"] = str(context.session.id)
    elif update.type == UpdateType.ASSISTANT_TOKEN:
        data["text"] = update.text
    elif update.type == UpdateType.ASSISTANT_MESSAGE:
        data["content"] = update.message.content
        message_dict = update.message.model_dump(by_alias=True)

        # Populate sender with full agent data if available
        if update.agent and message_dict.get("sender"):
            message_dict["sender"] = update.agent
            # Also add agent to top-level for debugging
            data["agent"] = update.agent

        data["message"] = message_dict
        if update.message.tool_calls:
            data["tool_calls"] = [
                dumps_json(tc.model_dump()) for tc in update.message.tool_calls
            ]
    elif update.type == UpdateType.USER_MESSAGE:
        # User messages should already have enriched sender data from add_chat_message
        data["message"] = (
            update.message.model_dump(by_alias=True)
            if hasattr(update.message, "model_dump")
            else update.message
        )
    elif update.type == UpdateType.TOOL_COMPLETE:
        data["tool"] = update.tool_name
        data["result"] = dumps_json(update.result)
    elif update.type == UpdateType.TOOL_CANCELLED:
        data["tool"] = update.tool_name
        data["tool_index"] = update.tool_index
        data["result"] = dumps_json(update.result)
    elif update.type == UpdateType.ERROR:
        data["error"] = update.error if hasattr(update, "error") else None
    elif update.type == UpdateType.END_PROMPT:
        pass

    return data


async def _run_single_actor_session(
    session: Session,
    context: PromptSessionContext,
    actor: Agent,
    *,
    stream: bool,
    is_client_platform: bool,
    session_run_id: str,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    stage_cm = (
        instrumentation.track_stage(f"actor:{actor.id}", level="info")
        if instrumentation
        else nullcontext()
    )
    with stage_cm:
        llm_context = await build_llm_context(
            session,
            actor,
            context,
            trace_id=session_run_id,
            instrumentation=instrumentation,
        )

        async for update in async_prompt_session(
            session,
            llm_context,
            actor,
            stream=stream,
            is_client_platform=is_client_platform,
            session_run_id=session_run_id,
            api_key_id=context.api_key_id,
            instrumentation=instrumentation,
        ):
            formatted_update = format_session_update(update, context)
            if instrumentation:
                instrumentation.log_update(
                    formatted_update.get("type", update.type.value),
                    formatted_update,
                )
            yield formatted_update


async def _run_multi_actor_sessions(
    session: Session,
    context: PromptSessionContext,
    actors: List[Agent],
    *,
    stream: bool,
    is_client_platform: bool,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    update_queue: asyncio.Queue = asyncio.Queue()
    tasks = []

    async def run_actor_session(actor: Agent):
        stage_cm = (
            instrumentation.track_stage(f"actor:{actor.id}", level="info")
            if instrumentation
            else nullcontext()
        )
        with stage_cm:
            try:
                actor_session_run_id = str(uuid.uuid4())
                llm_context = await build_llm_context(
                    session,
                    actor,
                    context,
                    trace_id=actor_session_run_id,
                    instrumentation=instrumentation,
                )
                async for update in async_prompt_session(
                    session,
                    llm_context,
                    actor,
                    stream=stream,
                    is_client_platform=is_client_platform,
                    session_run_id=actor_session_run_id,
                    api_key_id=context.api_key_id,
                    instrumentation=instrumentation,
                ):
                    formatted_update = format_session_update(update, context)
                    if instrumentation:
                        instrumentation.log_update(
                            formatted_update.get("type", update.type.value),
                            formatted_update,
                        )
                    await update_queue.put(formatted_update)
            except Exception as e:
                await update_queue.put(
                    {
                        "type": UpdateType.ERROR.value,
                        "error": str(e),
                        "actor_id": str(actor.id),
                        "update_config": context.update_config.model_dump()
                        if context.update_config
                        else None,
                    }
                )

    for actor in actors:
        tasks.append(asyncio.create_task(run_actor_session(actor)))

    try:
        completed_count = 0
        while completed_count < len(tasks):
            done_tasks = [t for t in tasks if t.done()]
            completed_count = len(done_tasks)
            try:
                update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                yield update
            except asyncio.TimeoutError:
                if completed_count == len(tasks):
                    break
                continue

        while not update_queue.empty():
            yield await update_queue.get()
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_prompt_session_internal(
    context: PromptSessionContext,
    background_tasks: BackgroundTasks,
    stream: bool = False,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    """Internal function that handles both streaming and non-streaming"""
    session = context.session
    debugger = instrumentation.debugger

    stage_cm = (
        instrumentation.track_stage("prompt_session.runtime", level="info")
        if instrumentation
        else nullcontext()
    )

    with stage_cm:
        try:
            debugger.log("Validating prompt session", emoji="info")
            validate_prompt_session(session, context)

            actors = await determine_actors(session, context)
            debugger.log(
                f"Found {len(actors)} actor(s)",
                {
                    "actors": [str(actor.id) for actor in actors] if actors else [],
                    "agent_count": len(session.agents) if session.agents else 0,
                },
                emoji="actor" if actors else "warning",
            )

            if instrumentation:
                instrumentation.set_gauge("actor_count", len(actors))
                if len(actors) == 1:
                    instrumentation.update_context(agent_id=str(actors[0].id))

            is_client_platform = context.update_config is not None

            if not actors:
                debugger.log(
                    "No actors found - session has no agents assigned",
                    level="warning",
                    emoji="warning",
                )
                debugger.end_section("_run_prompt_session_internal")
                return

            session_run_id = context.session_run_id or str(uuid.uuid4())
            context.session_run_id = session_run_id
            if instrumentation:
                instrumentation.update_context(session_run_id=session_run_id)
            debugger.log("Session run ID", {"id": session_run_id[:8]}, emoji="info")

            if context.update_config:
                debugger.log("Starting typing indicator", emoji="info")
                from eve.api.typing_coordinator import update_busy_state

                await update_busy_state(
                    context.update_config.model_dump()
                    if hasattr(context.update_config, "model_dump")
                    else context.update_config,
                    session_run_id,
                    True,
                )

            try:
                if len(actors) == 1:
                    async for update in _run_single_actor_session(
                        session,
                        context,
                        actors[0],
                        stream=stream,
                        is_client_platform=is_client_platform,
                        session_run_id=session_run_id,
                        instrumentation=instrumentation,
                    ):
                        yield update
                else:
                    async for update in _run_multi_actor_sessions(
                        session,
                        context,
                        actors,
                        stream=stream,
                        is_client_platform=is_client_platform,
                        instrumentation=instrumentation,
                    ):
                        yield update
            finally:
                if context.update_config:
                    from eve.api.typing_coordinator import update_busy_state

                    await update_busy_state(
                        context.update_config.model_dump()
                        if hasattr(context.update_config, "model_dump")
                        else context.update_config,
                        session_run_id,
                        False,
                    )

            if background_tasks:
                for actor in actors:
                    background_tasks.add_task(
                        memory_service.maybe_form_memories, actor.id, session, actor
                    )

                if (
                    context.notification_config
                    and context.notification_config.success_notification
                ):
                    background_tasks.add_task(
                        _send_session_notification,
                        context.notification_config,
                        session,
                        success=True,
                    )

        except Exception as e:
            if (
                background_tasks
                and context.notification_config
                and context.notification_config.failure_notification
            ):
                background_tasks.add_task(
                    _send_session_notification,
                    context.notification_config,
                    session,
                    success=False,
                    error=str(e),
                )
            raise
