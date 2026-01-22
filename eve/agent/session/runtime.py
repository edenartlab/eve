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
    EdenMessageData,
    EdenMessageType,
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
from eve.api.errors import APIError
from eve.task import Task
from eve.tool import Tool
from eve.user import Manna, Transaction, increment_message_count
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


def _resolve_triggering_user_id(
    context: PromptSessionContext,
) -> Optional[ObjectId]:
    """Resolve the user who triggered this session prompt."""
    if not context.acting_user_id and not context.initiating_user_id:
        return None
    return ObjectId(str(context.acting_user_id or context.initiating_user_id))


def _resolve_billed_user_id(
    context: PromptSessionContext,
    actor: Agent,
    is_client_platform: bool,
) -> Optional[ObjectId]:
    """
    Determine which user should be billed for a response.
    Defaults to the triggering user unless the agent owner sponsors usage.
    """
    owner_pays = getattr(actor, "owner_pays", "off") or "off"

    if owner_pays == "full":
        return actor.owner
    if owner_pays == "deployments" and is_client_platform:
        return actor.owner

    return _resolve_triggering_user_id(context)


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
        context: Optional[PromptSessionContext] = None,
    ):
        self.session = session
        self.llm_context = llm_context
        self.actor = actor
        self.stream = stream
        self.is_client_platform = is_client_platform
        self.session_run_id = session_run_id or str(uuid.uuid4())
        self.api_key_id = api_key_id
        self.instrumentation = instrumentation
        self.context = context
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

        # Resolve billing users
        self.triggering_user_id = (
            _resolve_triggering_user_id(context) if context else None
        )
        self.billed_user_id = (
            _resolve_billed_user_id(context, actor, is_client_platform)
            if context
            else None
        )
        # Default billing to billed_user (e.g., owner_pays). If none, fall back to
        # the triggering user who entered the prompt loop.
        self.billing_user_id = self.billed_user_id or self.triggering_user_id
        self.billed_user_doc = None
        self.billing_user_doc = None
        self.rate_limiter = None

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

            # Initialize billing docs and rate limiter if enabled
            if self.billing_user_id:
                from eve.user import User

                if not self.billing_user_doc:
                    self.billing_user_doc = User.from_mongo(self.billing_user_id)
                    if self.billing_user_doc and not self.billed_user_doc:
                        self.billed_user_doc = self.billing_user_doc

                if os.environ.get("FF_RATE_LIMITS") == "yes" and self.billing_user_doc:
                    from eve.api.rate_limiter import RateLimiter

                    self.rate_limiter = RateLimiter()

            prompt_session_finished = False
            while not prompt_session_finished:
                self._ensure_not_cancelled()
                await self._refresh_llm_messages()
                self._maybe_disable_tools()

                # Check rate limits before LLM call
                if self.rate_limiter and self.billing_user_doc:
                    try:
                        await self.rate_limiter.check_message_rate_limit(
                            self.billing_user_doc
                        )
                    except APIError as e:
                        rate_limit_message = self._persist_rate_limit_message(e)
                        yield SessionUpdate(
                            type=UpdateType.ASSISTANT_MESSAGE,
                            message=rate_limit_message,
                            session_run_id=self.session_run_id,
                        )
                        yield SessionUpdate(
                            type=UpdateType.END_PROMPT,
                            session_run_id=self.session_run_id,
                        )
                        return

                # Always attempt billing once rate limits are cleared (or disabled)
                if os.environ.get("FF_MANNA_BILLING"):
                    try:
                        self._charge_manna_for_message()
                    except APIError as e:
                        billing_error_message = self._persist_billing_error_message(e)
                        yield SessionUpdate(
                            type=UpdateType.ASSISTANT_MESSAGE,
                            message=billing_error_message,
                            session_run_id=self.session_run_id,
                        )
                        yield SessionUpdate(
                            type=UpdateType.END_PROMPT,
                            session_run_id=self.session_run_id,
                        )
                        return

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

                await self._maybe_notify_user(assistant_message)

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
        # Select messages including eden messages - all under the same limit
        # Eden messages are converted to user role with SystemMessage tags in convert_message_roles()
        fresh_messages = select_messages(self.session)

        system_message = self.llm_context.messages[0]
        pinned_messages = []

        # Preserve system messages AND pinned context messages (with <SystemMessage> tag)
        # These are created in-memory and not stored in the database, so they would be lost
        # if we only fetched from fresh_messages
        for msg in self.llm_context.messages[1:]:
            is_system_role = msg.role == "system"
            is_pinned_context = (
                msg.content
                and "<SystemMessage>" in msg.content
                and "</SystemMessage>" in msg.content
            )
            if is_system_role or is_pinned_context:
                pinned_messages.append(msg)
            else:
                break

        refreshed_messages = [system_message]
        if pinned_messages:
            refreshed_messages.extend(pinned_messages)
        refreshed_messages.extend(fresh_messages)
        refreshed_messages = label_message_channels(refreshed_messages, self.session)
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
            "llm_call_id": response.llm_call_id,
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
            session=[self.session.id],
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
            triggering_user=self.triggering_user_id,
            billed_user=self.billing_user_id,
            agent_owner=self.actor.owner,
            llm_call=llm_result.get("llm_call_id"),
        )
        assistant_message.save()

        # Increment message count for the agent (sender)
        increment_message_count(assistant_message.sender)

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

    def _persist_rate_limit_message(self, error: APIError) -> ChatMessage:
        """Persist a rate limit error message as an Eden message."""
        error_text = getattr(error, "detail", None) or str(error)
        eden_message = ChatMessage(
            session=[self.session.id],
            sender=ObjectId("000000000000000000000000"),
            role="eden",
            content=error_text,
            eden_message_data=EdenMessageData(
                message_type=EdenMessageType.RATE_LIMIT, error=error_text
            ),
            triggering_user=self.triggering_user_id,
            billed_user=self.billing_user_id,
            agent_owner=self.actor.owner,
        )
        eden_message.save()
        return eden_message

    def _persist_billing_error_message(self, error: APIError) -> ChatMessage:
        """Persist a billing/manna error as an Eden message."""
        error_text = getattr(error, "detail", None) or str(error)
        eden_message = ChatMessage(
            session=[self.session.id],
            sender=ObjectId("000000000000000000000000"),
            role="eden",
            content=error_text,
            triggering_user=self.triggering_user_id,
            billed_user=self.billing_user_id,
            agent_owner=self.actor.owner,
        )
        eden_message.save()
        return eden_message

    def _charge_manna_for_message(self, amount: float = 2):
        """Deduct manna from the billed user for a chat message."""
        user_doc = self.billing_user_doc or self.billed_user_doc
        if not user_doc and self.billing_user_id:
            from eve.user import User

            user_doc = User.from_mongo(self.billing_user_id)
            self.billing_user_doc = user_doc
            if user_doc and not self.billed_user_doc:
                self.billed_user_doc = user_doc

        if not user_doc:
            return

        if "free_tools" in (user_doc.featureFlags or []):
            return

        try:
            manna = Manna.load(user_doc.id)
            manna.spend(amount)
            Transaction(
                manna=manna.id,
                task=self.session.id,
                amount=-amount,
                type="spend",
            ).save()
            update_session_budget(self.session, manna_spent=amount)
        except Exception as e:
            raise APIError(f"Insufficient manna: {str(e)}", status_code=402)

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

    async def _maybe_notify_user(self, assistant_message: ChatMessage):
        try:
            is_active_response = await check_if_session_active(
                str(self.session.owner), str(self.session.id)
            )
            is_active = is_active_response.get("is_active", False)
            if not is_active:
                await create_session_message_notification(
                    user_id=str(self.session.owner),
                    session_id=str(self.session.id),
                    agent_id=str(self.actor.id) if self.actor else None,
                    message=assistant_message.content if assistant_message else None,
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
        """
        Setup Ably realtime listener for cancellation signals.

        This is non-critical functionality - if Ably fails to connect or disconnects,
        the session continues but loses remote cancellation capability.
        """
        try:
            from ably import AblyRealtime

            self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
            channel_name = f"{os.getenv('DB')}-session-cancel-{self.session.id}"
            channel = self.ably_client.channels.get(channel_name)

            # Monitor connection state - degrade gracefully on disconnect
            def on_connection_state_change(state_change):
                current_state = state_change.current
                if current_state in ("suspended", "failed", "closed"):
                    logger.warning(
                        f"Ably connection {current_state} for session {self.session.id}. "
                        f"Remote cancellation disabled - session will continue."
                    )
                    # Don't raise or block - just lose cancellation support
                elif current_state == "disconnected":
                    logger.info(
                        f"Ably disconnected for session {self.session.id}, attempting reconnect..."
                    )

            self.ably_client.connection.on(on_connection_state_change)

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

            # Use a timeout for subscription to prevent blocking on connection issues
            try:
                await asyncio.wait_for(
                    channel.subscribe("cancel", cancellation_handler),
                    timeout=10.0,  # 10 second timeout for initial subscription
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Ably subscription timed out for session {self.session.id}. "
                    f"Remote cancellation disabled - session will continue."
                )
                # Close the client to avoid zombie connections
                try:
                    await self.ably_client.close()
                except Exception:
                    pass
                self.ably_client = None

        except Exception as e:
            logger.warning(
                f"Failed to setup Ably cancellation for session {self.session.id}: {e}. "
                f"Remote cancellation disabled - session will continue."
            )
            # Ensure we don't leave a half-initialized client
            if self.ably_client:
                try:
                    await self.ably_client.close()
                except Exception:
                    pass
                self.ably_client = None

    async def _handle_session_cancelled(self):
        try:
            # Cancel any running tasks in this session
            await self._cancel_session_tasks()

            # Cancel ALL child sessions (from parent_session field)
            await self._cancel_all_child_sessions()

            last_message = self._fetch_last_message()
            if last_message and last_message.tool_calls:
                updated = False
                for idx, tool_call in enumerate(last_message.tool_calls):
                    # Cancel child session for any tool call that has one
                    # regardless of tool call status
                    if tool_call.child_session:
                        await self._cancel_child_session(tool_call.child_session)

                    if tool_call.status in ["pending", "running"]:
                        tool_call.status = "cancelled"
                        tool_call.result = [
                            {"status": "cancelled", "message": "Task cancelled by user"}
                        ]
                        updated = True

                        yield SessionUpdate(
                            type=UpdateType.TOOL_CANCELLED,
                            tool_name=tool_call.tool,
                            tool_index=idx,
                            result={
                                "status": "cancelled",
                                "message": "Task cancelled by user",
                            },
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
                session=[self.session.id],
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

    async def _cancel_all_child_sessions(self):
        """Cancel all child sessions by querying for sessions with parent_session field."""
        child_sessions = Session.find(
            {
                "parent_session": self.session.id,
                "status": {"$nin": ["archived", "finished"]},
            }
        )

        for child_session in child_sessions:
            try:
                await self._cancel_child_session(child_session.id)
            except Exception as e:
                logger.warning(
                    f"Failed to cancel child session {child_session.id}: {e}"
                )

    async def _cancel_session_tasks(self):
        """Cancel all pending/running tasks in this session."""
        tasks = Task.find(
            {"session": self.session.id, "status": {"$in": ["pending", "running"]}}
        )

        for task in tasks:
            try:
                tool = Tool.load(key=task.tool)
                await tool.async_cancel(task)
            except Exception as e:
                logger.warning(f"Failed to cancel task {task.id}: {e}")

    async def _cancel_child_session(self, child_session_id: ObjectId):
        """Cancel a child session by sending Ably signal and cancelling its tasks."""
        try:
            from ably import AblyRest

            child_session = Session.from_mongo(child_session_id)
            if not child_session:
                return

            # Send Ably cancel signal to child session
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel_name = f"{os.getenv('DB')}-session-cancel-{child_session_id}"
            channel = ably_client.channels.get(channel_name)

            await channel.publish(
                "cancel",
                {
                    "session_id": str(child_session_id),
                    "user_id": str(self.session.owner),
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
            )

            # Cancel tasks in child session
            child_tasks = Task.find(
                {"session": child_session_id, "status": {"$in": ["pending", "running"]}}
            )
            for task in child_tasks:
                try:
                    tool = Tool.load(key=task.tool)
                    await tool.async_cancel(task)
                except Exception as e:
                    logger.warning(f"Failed to cancel child task {task.id}: {e}")

        except Exception as e:
            logger.warning(f"Failed to cancel child session {child_session_id}: {e}")

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

        # Database fallback check (runs periodically, not every call)
        # This catches cancellations when Ably fails to deliver the signal
        if self._should_check_db_cancellation():
            if self._check_db_cancellation():
                self.cancellation_event.set()
                raise SessionCancelledException("Session cancelled by user")

    def _should_check_db_cancellation(self) -> bool:
        """Rate-limit database cancellation checks to avoid overhead."""
        import time

        if not hasattr(self, "_last_db_cancel_check"):
            self._last_db_cancel_check = 0

        now = time.time()
        # Check database every 2 seconds
        if now - self._last_db_cancel_check > 2.0:
            self._last_db_cancel_check = now
            return True
        return False

    def _check_db_cancellation(self) -> bool:
        """Check database for cancellation indicators.

        Note: We only check for the "Response cancelled by user" system message,
        NOT for cancelled tool calls. A cancelled tool call (subsession) should
        not cancel the parent session - only the child was cancelled.
        """
        try:
            # Check for cancellation message in session
            cancel_messages = ChatMessage.find(
                {
                    "session": self.session.id,
                    "content": "Response cancelled by user",
                    "role": "system",
                },
                limit=1,
            )
            if cancel_messages:
                logger.info(
                    f"Session {self.session.id} cancellation detected via database check"
                )
                return True

            return False
        except Exception as e:
            logger.warning(f"Error checking database for cancellation: {e}")
            return False

    async def _cleanup(self):
        self._remove_active_request()
        if self.ably_client:
            try:
                # Use timeout to prevent hanging if Ably connection is stuck
                await asyncio.wait_for(self.ably_client.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Ably client close timed out for session {self.session.id}"
                )
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
    context: Optional[PromptSessionContext] = None,
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
        context=context,
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


async def _run_actor_sessions(
    session: Session,
    context: PromptSessionContext,
    actors: List[Agent],
    *,
    stream: bool,
    is_client_platform: bool,
    session_run_id: str,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    """Run prompt sessions for one or more actors.

    Single actor: Fast path with direct yield
    Multiple actors: Parallel execution with queue-based update collection
    """
    if len(actors) == 1:
        # Fast path: single actor, direct yield
        async for update in _run_single_actor(
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
        # Multi-actor: parallel execution
        async for update in _run_multiple_actors(
            session,
            context,
            actors,
            stream=stream,
            is_client_platform=is_client_platform,
            instrumentation=instrumentation,
        ):
            yield update


async def _run_single_actor(
    session: Session,
    context: PromptSessionContext,
    actor: Agent,
    *,
    stream: bool,
    is_client_platform: bool,
    session_run_id: str,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    """Run prompt session for a single actor (fast path)."""
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
            context=context,
        ):
            formatted_update = format_session_update(update, context)
            if instrumentation:
                instrumentation.log_update(
                    formatted_update.get("type", update.type.value),
                    formatted_update,
                )
            yield formatted_update


async def _run_multiple_actors(
    session: Session,
    context: PromptSessionContext,
    actors: List[Agent],
    *,
    stream: bool,
    is_client_platform: bool,
    instrumentation: Optional[PromptSessionInstrumentation] = None,
):
    """Run prompt sessions for multiple actors in parallel."""
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
                    context=context,
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
    from loguru import logger

    session = context.session
    debugger = instrumentation.debugger if instrumentation else SessionDebugger()

    logger.info(f"[ORCH] _run_prompt_session_internal called for session {session.id}")
    logger.info(
        f"[ORCH] Session: type={session.session_type}, status={session.status}, agents={session.agents}"
    )
    logger.info(
        f"[ORCH] Context: initiating_user_id={context.initiating_user_id}, message={context.message}"
    )

    stage_cm = (
        instrumentation.track_stage("prompt_session.runtime", level="info")
        if instrumentation
        else nullcontext()
    )

    with stage_cm:
        try:
            debugger.log("Validating prompt session", emoji="info")
            logger.info("[ORCH] Validating prompt session")
            validate_prompt_session(session, context)
            logger.info("[ORCH] Validation passed")

            # Status check for automatic sessions - only skip if paused/archived
            # Note: "running" check is handled by automatic.py before it calls orchestration
            if session.session_type == "automatic":
                logger.info(f"[ORCH] Automatic session, status={session.status}")
                if session.status in ("paused", "archived"):
                    debugger.log(
                        f"Session status is '{session.status}', skipping orchestration",
                        level="info",
                        emoji="info",
                    )
                    logger.info(f"[ORCH] Skipping - session is {session.status}")
                    return

            logger.info("[ORCH] Calling determine_actors")
            actors = await determine_actors(session, context)
            logger.info(
                f"[ORCH] determine_actors returned {len(actors) if actors else 0} actors: {[a.username for a in actors] if actors else []}"
            )
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
                async for update in _run_actor_sessions(
                    session,
                    context,
                    actors,
                    stream=stream,
                    is_client_platform=is_client_platform,
                    session_run_id=session_run_id,
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
