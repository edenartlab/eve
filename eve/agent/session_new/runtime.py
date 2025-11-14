import asyncio
import datetime
import json
import os
from time import timezone
import traceback
import uuid
from typing import Optional

from bson import ObjectId
from fastapi import BackgroundTasks
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.memory.memory_models import select_messages
from eve.agent.memory.service import memory_service
from eve.agent.session.debug_logger import SessionDebugger
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageObservability,
    LLMContext,
    PromptSessionContext,
    Session,
    SessionMemoryContext,
    SessionUpdate,
    ToolCall,
    UpdateType,
)
from eve.agent.llm.llm import (
    async_prompt as provider_async_prompt,
    async_prompt_stream as provider_async_prompt_stream,
    get_provider,
)
from eve.agent.session.session_llm import (
    async_prompt as legacy_async_prompt,
    async_prompt_stream as legacy_async_prompt_stream,
)
from eve.agent.session.tracing import trace_async_operation
from eve.api.errors import handle_errors
from eve.api.helpers import emit_update
from eve.utils import dumps_json

from .budget import update_session_budget, validate_prompt_session
from .context import (
    add_chat_message,
    build_llm_context,
    convert_message_roles,
    determine_actors,
    label_message_channels,
)
from .notifications import (
    _send_session_notification,
    check_if_session_active,
    create_session_message_notification,
)
from .tools import process_tool_calls


class SessionCancelledException(Exception):
    """Exception raised when a session is cancelled via Ably signal."""


async def async_prompt_session(
    session: Session,
    llm_context: LLMContext,
    actor: Agent,
    stream: bool = False,
    is_client_platform: bool = False,
    session_run_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
):
    debugger = SessionDebugger(str(session.id) if session else None)
    # Generate session_run_id if not provided to prevent None from being added to active_requests
    if session_run_id is None:
        session_run_id = str(uuid.uuid4())

    # Start distributed tracing transaction and set as active in scope
    import sentry_sdk

    transaction = sentry_sdk.start_transaction(
        name="prompt_session",
        op="session.prompt",
    )

    # Set tags on the transaction
    if transaction:
        transaction.set_tag("session_id", str(session.id))
        if llm_context.metadata and llm_context.metadata.trace_metadata:
            transaction.set_tag(
                "user_id", str(llm_context.metadata.trace_metadata.user_id)
            )
        transaction.set_tag("agent_id", str(actor.id))
        transaction.set_tag("session_run_id", session_run_id)
        transaction.set_tag("stream", str(stream))

        # Set as active span (correct API - direct assignment, not set_span())
        sentry_sdk.Hub.current.scope.span = transaction

    # Set up cancellation handling via Ably
    cancellation_event = asyncio.Event()
    tool_cancellation_events = {}  # Dict to track individual tool cancellations
    ably_client = None

    try:
        from ably import AblyRealtime

        ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        channel_name = f"{os.getenv('DB')}-session-cancel-{session.id}"
        channel = ably_client.channels.get(channel_name)

        async def cancellation_handler(message):
            """Handle cancellation messages from Ably."""
            try:
                data = message.data
                if isinstance(data, dict) and data.get("session_id") == str(session.id):
                    # Check if this is a tool-specific cancellation
                    tool_call_id = data.get("tool_call_id")

                    if tool_call_id is not None:
                        # Cancel specific tool call
                        if tool_call_id in tool_cancellation_events:
                            tool_cancellation_events[tool_call_id].set()
                    else:
                        # Check if this is a trace-specific cancellation
                        cancel_trace_id = data.get("trace_id")
                        if cancel_trace_id is None or cancel_trace_id == session_run_id:
                            # Cancel if no specific trace_id is provided (cancel all)
                            # or if the trace_id matches this session_run_id
                            cancellation_event.set()
            except Exception as e:
                logger.error(f"Error in cancellation handler: {e}")

        await channel.subscribe("cancel", cancellation_handler)

    except Exception as e:
        logger.error(f"Failed to setup Ably cancellation for session {session.id}: {e}")
        # Continue without cancellation support if Ably fails

    async def prompt_session_generator():
        """Generator function that yields session updates and can be cancelled."""
        active_requests = session.active_requests or []
        active_requests.append(session_run_id)
        # session.active_requests = active_requests
        # session.save()
        session.update(active_requests=active_requests)

        yield SessionUpdate(
            type=UpdateType.START_PROMPT,
            agent={
                "_id": str(actor.id),
                "username": actor.username,
                "name": actor.name,
                "userImage": actor.userImage,
            }
            if actor
            else None,
            session_run_id=session_run_id,
        )

        prompt_session_finished = False
        tokens_spent = 0
        tool_was_cancelled = False  # Track if any tool was cancelled

        while not prompt_session_finished:
            # Check for cancellation before each iteration
            if cancellation_event.is_set():
                raise SessionCancelledException("Session cancelled by user")

            # Refresh messages from database to get any new messages added during tool calls
            # This ensures we have the latest context including any user messages sent while tools were running
            fresh_messages = select_messages(session)

            # Rebuild the messages list with fresh data
            system_message = llm_context.messages[0]  # Keep the system message
            system_extras = []
            # Extract any system extras (messages with role="system" after the first one)
            for msg in llm_context.messages[1:]:
                if msg.role == "system":
                    system_extras.append(msg)
                else:
                    break

            if session.trigger:
                from eve.trigger import Trigger

                trigger = Trigger.from_mongo(session.trigger)
                trigger_message = ChatMessage(
                    session=session.id,
                    role="system",
                    content=f"<Full Task Context>\n{trigger.context}\n</Full Task Context>",
                )
                system_extras.append(trigger_message)

            # Rebuild messages with fresh data from database
            refreshed_messages = [system_message]
            if system_extras:
                refreshed_messages.extend(system_extras)
            refreshed_messages.extend(fresh_messages)
            refreshed_messages = label_message_channels(refreshed_messages)
            refreshed_messages = convert_message_roles(refreshed_messages, actor.id)

            # Update the context with refreshed messages
            llm_context.messages = refreshed_messages

            # Generate new generation_id for this LLM call
            llm_context.metadata.generation_id = str(uuid.uuid4())

            # Disable tools if any tool was cancelled in this session run
            if tool_was_cancelled:
                llm_context.tools = {}
                llm_context.tool_choice = "none"

            provider = get_provider(llm_context)
            provider_label = (
                provider.__class__.__name__ if provider else "LegacyLiteLLM"
            )
            debugger.log(
                "Selected LLM provider",
                {"provider": provider_label},
                emoji="llm",
            )

            if stream:
                # For streaming, we need to collect the content as it comes in
                content = ""
                tool_calls_dict = {}  # Track tool calls by index to accumulate arguments
                stop_reason = None
                tokens_spent = 0  # Initialize tokens_spent for streaming

                async with trace_async_operation(
                    "llm.stream", model=llm_context.config.model
                ):
                    stream_iter = (
                        provider_async_prompt_stream(llm_context, provider)
                        if provider
                        else legacy_async_prompt_stream(llm_context)
                    )
                    async for chunk in stream_iter:
                        # Check for cancellation during streaming
                        if cancellation_event.is_set():
                            raise SessionCancelledException("Session cancelled by user")

                        if hasattr(chunk, "choices") and chunk.choices:
                            choice = chunk.choices[0]
                            # Only yield content tokens, not tool call chunks
                            if choice.delta and choice.delta.content:
                                content += choice.delta.content
                                yield SessionUpdate(
                                    type=UpdateType.ASSISTANT_TOKEN,
                                    text=choice.delta.content,
                                    session_run_id=session_run_id,
                                )
                            # Process tool calls silently (don't yield anything)
                            if choice.delta and choice.delta.tool_calls:
                                for tc in choice.delta.tool_calls:
                                    if tc.index not in tool_calls_dict:
                                        tool_calls_dict[tc.index] = {
                                            "id": tc.id,
                                            "name": tc.function.name
                                            if tc.function
                                            else None,
                                            "arguments": "",
                                        }
                                    if tc.function and tc.function.arguments:
                                        tool_calls_dict[tc.index]["arguments"] += (
                                            tc.function.arguments
                                        )
                            if choice.finish_reason:
                                stop_reason = choice.finish_reason

                        # Capture token usage from streaming response
                        if hasattr(chunk, "usage") and chunk.usage:
                            tokens_spent = chunk.usage.total_tokens

                # Convert accumulated tool calls to ToolCall objects
                tool_calls = None
                if tool_calls_dict:
                    tool_calls = []
                    for idx in sorted(tool_calls_dict.keys()):
                        tc_data = tool_calls_dict[idx]
                        try:
                            args = (
                                json.loads(tc_data["arguments"])
                                if tc_data["arguments"]
                                else {}
                            )
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

                # Create the final assistant message
                assistant_message = ChatMessage(
                    session=session.id,
                    sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=stop_reason,
                    llm_config=llm_context.config.__dict__
                    if llm_context.config
                    else None,
                    observability=ChatMessageObservability(
                        session_id=llm_context.metadata.session_id,
                        trace_id=llm_context.metadata.trace_id,
                        generation_id=llm_context.metadata.generation_id,
                        tokens_spent=tokens_spent,
                    ),
                    apiKey=ObjectId(api_key_id) if api_key_id else None,
                )
            else:
                # Non-streaming path
                async with trace_async_operation(
                    "llm.prompt", model=llm_context.config.model
                ):
                    if provider:
                        response = await provider_async_prompt(llm_context, provider)
                    else:
                        response = await legacy_async_prompt(llm_context)
                assistant_message = ChatMessage(
                    session=session.id,
                    sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    finish_reason=response.stop,
                    thought=response.thought,
                    llm_config=llm_context.config.__dict__
                    if llm_context.config
                    else None,
                    observability=ChatMessageObservability(
                        session_id=llm_context.metadata.session_id,
                        trace_id=llm_context.metadata.trace_id,
                        generation_id=llm_context.metadata.generation_id,
                        tokens_spent=response.tokens_spent,
                    ),
                    apiKey=ObjectId(api_key_id) if api_key_id else None,
                )
                stop_reason = response.stop
                tokens_spent = response.tokens_spent

            assistant_message.save()

            # increment agent stats
            # stats = actor.stats
            # stats["messageCount"] += 1
            # actor.update(stats=stats.model_dump())

            # No longer storing message IDs on session to avoid race conditions
            # session.messages.append(assistant_message.id)

            # session.memory_context.last_activity = datetime.now(timezone.utc)
            # session.memory_context.messages_since_memory_formation += 1
            memory_context = session.memory_context
            memory_context.last_activity = datetime.now(timezone.utc)
            memory_context.messages_since_memory_formation += 1
            session.update(memory_context=memory_context.model_dump())
            session.memory_context = SessionMemoryContext(**session.memory_context)

            update_session_budget(session, tokens_spent=tokens_spent, turns_spent=1)

            # Add tracing data and capture Sentry trace ID
            if transaction:
                transaction.set_data("tokens_spent", tokens_spent)
                transaction.set_data("stop_reason", stop_reason)
                if assistant_message.tool_calls:
                    transaction.set_data(
                        "tool_calls_count", len(assistant_message.tool_calls)
                    )

                # Capture Sentry trace ID for cross-system correlation
                if assistant_message.observability and hasattr(transaction, "trace_id"):
                    assistant_message.observability.sentry_trace_id = (
                        transaction.trace_id
                    )
                    assistant_message.save()  # Update with Sentry trace ID

            # No longer appending to llm_context.messages since we refresh from DB each iteration
            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE,
                message=assistant_message,
                agent={
                    "_id": str(actor.id),
                    "username": actor.username,
                    "name": actor.name,
                    "userImage": actor.userImage,
                },
                session_run_id=session_run_id,
            )

            # Create notification if user is not actively viewing the session
            # This runs in the background and won't block message delivery
            try:
                is_active_response = await check_if_session_active(
                    str(session.owner), str(session.id)
                )

                is_active = is_active_response.get("is_active", False)

                # Only create notification if user is NOT viewing
                if not is_active:
                    await create_session_message_notification(
                        user_id=str(session.owner),
                        session_id=str(session.id),
                        agent_id=str(actor.id),
                    )
            except Exception as e:
                logger.warning(
                    f"[NOTIFICATION] ❌ Failed to create session message notification: {e}"
                )

            if assistant_message.tool_calls:
                async with trace_async_operation(
                    "tools.process_all", tool_count=len(assistant_message.tool_calls)
                ):
                    async for update in process_tool_calls(
                        session,
                        assistant_message,
                        llm_context,
                        cancellation_event,
                        tool_cancellation_events,
                        is_client_platform,
                        session_run_id,
                    ):
                        # Check for cancellation during tool execution
                        if cancellation_event.is_set():
                            raise SessionCancelledException("Session cancelled by user")
                        # Track if any tool was cancelled
                        if update.type == UpdateType.TOOL_CANCELLED:
                            tool_was_cancelled = True
                        yield update

            if stop_reason in ["stop", "completed"]:
                prompt_session_finished = True

            # if tool choice was previously set to something specific, set it to None for follow-up messages
            llm_context.tool_choice = "auto"

        yield SessionUpdate(
            type=UpdateType.END_PROMPT,
            session_run_id=session_run_id,
        )

    try:
        # Run the prompt session generator, checking for cancellation
        async for update in prompt_session_generator():
            yield update

    except SessionCancelledException:
        # Handle graceful cancellation
        try:
            # 1. Mark any unfinished tool calls as cancelled
            last_message = None
            last_messages = ChatMessage.find(
                {"session": session.id},
                sort="createdAt",
                desc=True,
                limit=1,
            )
            if last_messages:
                last_message = last_messages[0]

            if last_message and last_message.tool_calls:
                for idx, tool_call in enumerate(last_message.tool_calls):
                    if tool_call.status in ["pending", "running"]:
                        tool_call.status = "cancelled"
                        # Yield cancellation update for each tool call
                        yield SessionUpdate(
                            type=UpdateType.TOOL_CANCELLED,
                            tool_name=tool_call.tool,
                            tool_index=idx,
                            result={"status": "cancelled"},
                            session_run_id=session_run_id,
                        )

                # Force save by updating the entire tool_calls array
                try:
                    # Save using direct MongoDB update to ensure the change persists
                    messages_collection = ChatMessage.get_collection()
                    messages_collection.update_one(
                        {"_id": last_message.id},
                        {
                            "$set": {
                                "tool_calls": [
                                    tc.model_dump() for tc in last_message.tool_calls
                                ]
                            }
                        },
                    )
                except Exception:
                    last_message.save()

            # 2. Add system message indicating cancellation
            cancel_message = ChatMessage(
                session=session.id,
                sender=ObjectId("000000000000000000000000"),  # System sender
                role="system",
                content="Response cancelled by user",
            )
            cancel_message.save()
            # No longer storing message IDs on session to avoid race conditions
            # session.messages.append(cancel_message.id)

            # 3. Yield final updates
            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE,
                message=cancel_message,
                session_run_id=session_run_id,
            )
            yield SessionUpdate(
                type=UpdateType.END_PROMPT, session_run_id=session_run_id
            )

        except Exception as e:
            logger.error(f"Error during session cancellation cleanup: {e}")
            yield SessionUpdate(
                type=UpdateType.END_PROMPT, session_run_id=session_run_id
            )

    finally:
        active_requests = session.active_requests or []
        active_requests.remove(session_run_id)
        # session.active_requests = active_requests
        # session.save()
        session.update(active_requests=active_requests)

        # Clean up Ably subscription
        if ably_client:
            try:
                await ably_client.close()
            except Exception as e:
                logger.error(f"Error closing Ably client: {e}")

        # Finish transaction
        if transaction:
            transaction.finish()


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


async def _run_prompt_session_internal(
    context: PromptSessionContext,
    background_tasks: BackgroundTasks,
    stream: bool = False,
):
    """Internal function that handles both streaming and non-streaming"""
    session = context.session
    session_id = str(session.id) if session else None
    debugger = SessionDebugger(session_id)

    try:
        debugger.log("Validating prompt session", emoji="info")
        validate_prompt_session(session, context)

        # Create user message first, regardless of whether actors are determined
        if context.initiating_user_id:
            await add_chat_message(session, context)

        actors = await determine_actors(session, context)
        debugger.log(
            f"Found {len(actors)} actor(s)",
            {
                "actors": [str(actor.id)[:8] for actor in actors] if actors else [],
                "agent_count": len(session.agents) if session.agents else 0,
            },
            emoji="actor" if actors else "warning",
        )

        is_client_platform = context.update_config is not None

        if not actors:
            debugger.log(
                "No actors found - session has no agents assigned",
                level="warning",
                emoji="warning",
            )
            debugger.end_section("_run_prompt_session_internal")
            return

        # Generate session run ID for this prompt session
        session_run_id = str(uuid.uuid4())
        debugger.log("Session run ID", {"id": session_run_id[:8]}, emoji="info")

        # Start typing indicator
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
            # For single actor, maintain backwards compatibility
            if len(actors) == 1:
                actor = actors[0]

                debugger.log("Building LLM context", emoji="llm")
                llm_context = await build_llm_context(
                    session,
                    actor,
                    context,
                    trace_id=session_run_id,
                )

                debugger.log("Starting prompt session", emoji="llm")
                async for update in async_prompt_session(
                    session,
                    llm_context,
                    actor,
                    stream=stream,
                    is_client_platform=is_client_platform,
                    session_run_id=session_run_id,
                    api_key_id=context.api_key_id,
                ):
                    formatted_update = format_session_update(update, context)
                    debugger.log_update(
                        update.type.value if hasattr(update, "type") else "unknown",
                        formatted_update,
                    )
                    yield formatted_update
            else:
                # Multiple actors - run them in parallel with streaming
                update_queue = asyncio.Queue()
                tasks = []

                async def run_actor_session(actor: Agent, queue: asyncio.Queue):
                    try:
                        # Each actor gets its own generation ID
                        actor_session_run_id = str(uuid.uuid4())
                        llm_context = await build_llm_context(
                            session,
                            actor,
                            context,
                            trace_id=actor_session_run_id,
                        )
                        async for update in async_prompt_session(
                            session,
                            llm_context,
                            actor,
                            stream=stream,
                            is_client_platform=is_client_platform,
                            session_run_id=actor_session_run_id,
                            api_key_id=context.api_key_id,
                        ):
                            formatted_update = format_session_update(update, context)
                            await queue.put(formatted_update)
                    except Exception as e:
                        await queue.put(
                            {
                                "type": UpdateType.ERROR.value,
                                "error": str(e),
                                "actor_id": str(actor.id),
                                "update_config": context.update_config.model_dump()
                                if context.update_config
                                else None,
                            }
                        )

                # Create tasks for all actors
                for actor in actors:
                    task = asyncio.create_task(run_actor_session(actor, update_queue))
                    tasks.append(task)

                # Yield updates as they arrive from any actor
                completed_count = 0
                while completed_count < len(tasks):
                    # Check if any tasks are done
                    done_tasks = [t for t in tasks if t.done()]
                    completed_count = len(done_tasks)

                    try:
                        # Get update with timeout to periodically check task status
                        update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                        yield update
                    except asyncio.TimeoutError:
                        # No update available, check if all tasks are done
                        if completed_count == len(tasks):
                            break
                        continue

                # Drain any remaining updates from the queue
                while not update_queue.empty():
                    update = await update_queue.get()
                    yield update

                # Ensure all tasks are complete
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Stop typing indicator
            if context.update_config:
                await update_busy_state(
                    context.update_config.model_dump()
                    if hasattr(context.update_config, "model_dump")
                    else context.update_config,
                    session_run_id,
                    False,
                )

        # Schedule background tasks if available
        if background_tasks:
            # Process memory formation for all actors that participated
            for actor in actors:
                background_tasks.add_task(
                    memory_service.maybe_form_memories, actor.id, session, actor
                )

            # Send success notification if configured
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
        # Send failure notification if configured
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
        # Re-raise the exception
        raise

