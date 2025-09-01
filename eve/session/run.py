import os
import uuid
import logging
import asyncio
from enum import Enum
from bson import ObjectId
from ably import AblyRealtime
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from sentry_sdk import capture_exception
from pydantic import ConfigDict, BaseModel

from eve.mongo import get_collection
from eve.agent import Agent
from eve.session.session import Session
from eve.session.message import ToolCall, ChatMessage, ChatMessageObservability
from eve.llm.llm import LLMContext, async_prompt


logger = logging.getLogger(__name__)


class UpdateType(Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    TOOL_CANCELLED = "tool_cancelled"
    ERROR = "error"
    END_PROMPT = "end_prompt"


class SessionUpdateConfig(BaseModel):
    """Config for updating the session"""

    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    deployment_id: Optional[str] = None
    discord_channel_id: Optional[str] = None
    discord_message_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_message_id: Optional[str] = None
    telegram_thread_id: Optional[str] = None
    farcaster_hash: Optional[str] = None
    farcaster_author_fid: Optional[int] = None
    farcaster_message_id: Optional[str] = None
    twitter_tweet_to_reply_id: Optional[str] = None
    user_is_bot: Optional[bool] = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionUpdate(BaseModel):
    type: UpdateType
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    update_config: Optional[SessionUpdateConfig] = None
    agent: Optional[Dict[str, Any]] = None
    session_run_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionCancelledException(Exception):
    """Exception raised when a session is cancelled via Ably signal."""
    pass


async def async_prompt_session(
    session: Session,
    llm_context: LLMContext,
    actor: Agent,
    stream: bool = False,
    is_client_platform: bool = False,
    session_run_id: Optional[str] = None,
):
    """Entrypoint for advancing a session forward (generating an LLM response)."""
    session_run_id = session_run_id or str(uuid.uuid4())

    # --- cancellation setup (Ably + local events) ---
    cancellation_event = asyncio.Event()
    tool_cancellation_events: Dict[str, asyncio.Event] = {}
    ably_client = None
    try:
        ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        cancel_channel_name = f"{os.getenv('DB')}-session-cancel-{session.id}"
        channel = ably_client.channels.get(cancel_channel_name)

        async def on_cancel(msg):
            data = msg.data
            if not isinstance(data, dict) or data.get("session_id") != str(session.id):
                return
            tc_id = data.get("tool_call_id")
            trace_id = data.get("trace_id")
            if tc_id:
                tool_cancellation_events.setdefault(tc_id, asyncio.Event()).set()
            elif trace_id is None or trace_id == session_run_id:
                cancellation_event.set()

        await channel.subscribe("cancel", on_cancel)
    except Exception as e:
        # If Ably is unavailable, session won't be cancelable remotely.
        logger.error(f"Failed to setup Ably cancellation for session {session.id}: {e}")

    # --- mark active, announce start ---
    active = session.active_requests or []
    active.append(session_run_id)
    session.active_requests = active
    session.save()

    yield SessionUpdate(
        type=UpdateType.START_PROMPT,
        agent={
            "_id": str(actor.id),
            "username": actor.username,
            "name": actor.name,
            "userImage": actor.userImage,
        },
        session_run_id=session_run_id,
    )

    try:
        finished = False
        while not finished:
            if cancellation_event.is_set():
                raise SessionCancelledException("Session cancelled by user")

            # new generation id for each LLM call
            llm_context.metadata.generation_id = str(uuid.uuid4())

            # --- get LLM output (streaming or not) ---
            tokens_spent = 0
            stop_reason = None
            assistant_message: ChatMessage

            response = await async_prompt(llm_context)
            tokens_spent = response.tokens_spent
            stop_reason = response.stop

            observability = ChatMessageObservability(
                session_id=llm_context.metadata.session_id,
                trace_id=llm_context.metadata.trace_id,
                generation_id=llm_context.metadata.generation_id,
                tokens_spent=tokens_spent,
            )

            assistant_message = ChatMessage(
                session=session.id,
                role="assistant",
                sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
                thought=response.thought,
                content=response.content,
                tool_calls=response.tool_calls,
                finish_reason=response.stop,
                llm_config=llm_context.config.__dict__,
                observability=observability,
            )

            # --- save assistant message ---
            assistant_message.save()

            # --- update session ---
            session.messages.append(assistant_message.id)
            session.memory_context.last_activity = datetime.now(timezone.utc)
            session.memory_context.messages_since_memory_formation += 1
            update_session_budget(session, tokens_spent=tokens_spent, turns_spent=1)
            session.save()

            # --- update context and yield update ---
            llm_context.messages.append(assistant_message)

            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE, 
                message=assistant_message
            )

            # --- tools (with per-tool cancellation) ---
            if assistant_message.tool_calls:
                async for update in process_tool_calls(
                    session=session,
                    assistant_message=assistant_message,
                    llm_context=llm_context,
                    cancellation_event=cancellation_event,
                    tool_cancellation_events=tool_cancellation_events,
                    is_client_platform=is_client_platform,
                ):
                    if cancellation_event.is_set():
                        raise SessionCancelledException("Session cancelled by user")
                    yield update

            finished = stop_reason.lower() in ("stop", "completed")

        yield SessionUpdate(type=UpdateType.END_PROMPT, session_run_id=session_run_id)

    except SessionCancelledException:
        # cancel any running/pending tool calls on the last message
        last_message = None
        if session.messages:
            last_message = ChatMessage.from_mongo(session.messages[-1])
            if last_message and last_message.tool_calls:
                for i, tc in enumerate(last_message.tool_calls):
                    if tc.status in ("pending", "running"):
                        tc.status = "cancelled"
                        yield SessionUpdate(
                            type=UpdateType.TOOL_CANCELLED,
                            tool_name=tc.tool,
                            tool_index=i,
                            result={"status": "cancelled"},
                        )
                # force-save tool calls
                try:
                    get_collection("messages").update_one(
                        {"_id": last_message.id},
                        {"$set": {"tool_calls": [tc.model_dump() for tc in last_message.tool_calls]}},
                    )
                except Exception:
                    last_message.markModified("tool_calls")
                    last_message.save()

        cancel_msg = ChatMessage(
            session=session.id,
            role="system",
            content="Response cancelled by user",
        )
        cancel_msg.save()
        session.messages.append(cancel_msg.id)
        
        yield SessionUpdate(type=UpdateType.ASSISTANT_MESSAGE, message=cancel_msg)
        yield SessionUpdate(type=UpdateType.END_PROMPT, session_run_id=session_run_id)

    finally:
        # clear active flag
        active = session.active_requests or []
        if session_run_id in active:
            active.remove(session_run_id)
        session.active_requests = active
        session.save()

        if ably_client:
            try:
                await ably_client.close()
            except Exception:
                pass


async def process_tool_call(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    tool_call: ToolCall,
    tool_call_index: int,
    cancellation_event: asyncio.Event = None,
    tool_cancellation_event: asyncio.Event = None,
    is_client_platform: bool = False,
):
    # convenience refs
    global_cancel = cancellation_event and cancellation_event.is_set()
    this_cancel = tool_cancellation_event and tool_cancellation_event.is_set()
    effective_cancel = tool_cancellation_event or cancellation_event

    # fast DB sync of one tool_call slot
    def sync_tool_call():
        try:
            get_collection("messages").update_one(
                {"_id": assistant_message.id},
                {"$set": {f"tool_calls.{tool_call_index}": assistant_message.tool_calls[tool_call_index].model_dump()}},
            )
        except Exception:
            assistant_message.markModified("tool_calls")
            assistant_message.save()

    # early cancellation
    if global_cancel or this_cancel:
        tool_call.status = "cancelled"
        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "cancelled"
            sync_tool_call()
        return SessionUpdate(
            type=UpdateType.TOOL_CANCELLED,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            result={"status": "cancelled"},
        )

    # mark running
    tool_call.status = "running"
    if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
        tc = assistant_message.tool_calls[tool_call_index]
        tc.status = "running"
        sync_tool_call()

    # execute
    try:
        result = await async_run_tool_call_with_cancellation(
            llm_context,
            tool_call,
            user_id=llm_context.metadata.trace_metadata.user_id
                or llm_context.metadata.trace_metadata.agent_id,
            agent_id=llm_context.metadata.trace_metadata.agent_id,
            cancellation_event=effective_cancel,
            is_client_platform=is_client_platform,
        )
    except Exception as e:
        capture_exception(e)
        tool_call.status = "failed"
        tool_call.error = str(e)
        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "failed"
            tc.error = str(e)
            sync_tool_call()
        return SessionUpdate(
            type=UpdateType.ERROR,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            error=str(e),
        )

    # cancellation after execution
    if cancellation_event and cancellation_event.is_set():
        tool_call.status = "cancelled"
        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "cancelled"
            sync_tool_call()
        return SessionUpdate(
            type=UpdateType.TOOL_CANCELLED,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            result={"status": "cancelled"},
        )

    # normalize non-dict result (e.g., special-cased tools) as completed
    if not isinstance(result, dict):
        tool_call.status = "completed"
        tool_call.result = result
        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "completed"
            tc.result = result
            tc.cost = result.get("cost", 0)
            tc.task = result.get("task")
            sync_tool_call()
        return SessionUpdate(
            type=UpdateType.TOOL_COMPLETE,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            result={"status": "completed", "result": result, "cost": None, "task": None},
        )

    status = result.get("status")
    if status == "completed":
        tool_result = result.get("result", [])
        tool_call.status = "completed"
        tool_call.result = tool_result

        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "completed"
            tc.result = tool_result
            tc.cost = result.get("cost", 0)
            tc.task = result.get("task")
            sync_tool_call()

        update_session_budget(session, manna_spent=result.get("cost", 0))
        session.save()

        return SessionUpdate(
            type=UpdateType.TOOL_COMPLETE,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            result=result,
        )

    if status == "cancelled":
        tool_call.status = "cancelled"
        if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
            tc = assistant_message.tool_calls[tool_call_index]
            tc.status = "cancelled"
            sync_tool_call()
        return SessionUpdate(
            type=UpdateType.TOOL_CANCELLED,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            result={"status": "cancelled"},
        )

    # failed / unknown
    err = result.get("error")
    tool_call.status = "failed"
    tool_call.error = err
    if assistant_message.tool_calls and tool_call_index < len(assistant_message.tool_calls):
        tc = assistant_message.tool_calls[tool_call_index]
        tc.status = "failed"
        tc.error = err
        sync_tool_call()
    return SessionUpdate(
        type=UpdateType.ERROR,
        tool_name=tool_call.tool,
        tool_index=tool_call_index,
        error=err,
    )


async def process_tool_calls(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    cancellation_event: asyncio.Event = None,
    tool_cancellation_events: dict = None,
    is_client_platform: bool = False,
):
    tool_calls = assistant_message.tool_calls or []
    tool_cancellation_events = tool_cancellation_events or {}

    # cap concurrency to 4 (same as old batching behavior)
    sem = asyncio.Semaphore(4)

    async def run_one(idx: int, tc: ToolCall):
        tool_cancellation_events.setdefault(tc.id, asyncio.Event())
        async with sem:
            return await process_tool_call(
                session=session,
                assistant_message=assistant_message,
                llm_context=llm_context,
                tool_call=tc,
                tool_call_index=idx,
                cancellation_event=cancellation_event,
                tool_cancellation_event=tool_cancellation_events[tc.id],
                is_client_platform=is_client_platform,
            )

    tasks = [
        asyncio.create_task(run_one(i, tc))
        for i, tc in enumerate(tool_calls)
        if tc.status != "completed"
    ]
    if not tasks:
        return

    for future in asyncio.as_completed(tasks):
        result = await future
        yield result


async def async_run_tool_call_with_cancellation(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
    cancellation_event: asyncio.Event = None,
):
    # passthrough for synthetic "web_search" results
    if tool_call.tool == "web_search":
        return tool_call.result

    tool = llm_context.tools.get(tool_call.tool)
    if not tool:
        return {"status": "failed", "error": f"Unknown tool: {tool_call.tool}"}

    if cancellation_event and cancellation_event.is_set():
        return {"status": "cancelled", "error": "Task cancelled by user"}

    task = await tool.async_start_task(
        user_id=user_id,
        agent_id=agent_id,
        args=tool_call.args,
        mock=False,
        public=public,
        is_client_platform=is_client_platform,
    )

    if not cancellation_event:
        result = await tool.async_wait(task)
    else:
        wait_future = asyncio.create_task(tool.async_wait(task))
        cancel_future = asyncio.create_task(cancellation_event.wait())
        try:
            done, _ = await asyncio.wait({wait_future, cancel_future}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_future in done:
                try:
                    await tool.async_cancel(task)
                except Exception:
                    pass
                result = {"status": "cancelled", "error": "Task cancelled by user"}
            else:
                result = await wait_future
        finally:
            # cleanup pending futures
            for future in (wait_future, cancel_future):
                if not future.done():
                    future.cancel()
                    try:
                        await future
                    except asyncio.CancelledError:
                        pass

    # attach accounting if result is a dict; keep non-dict as-is for callers that expect it
    if isinstance(result, dict):
        result.setdefault("cost", getattr(task, "cost", None))
        result.setdefault("task", getattr(task, "id", None))
    return result


def update_session_budget(
    session: Session,
    tokens_spent: Optional[int] = None,
    manna_spent: Optional[float] = None,
    turns_spent: Optional[int] = None,
):
    if session.budget:
        if tokens_spent:
            session.budget.tokens_spent += tokens_spent
        if manna_spent:
            session.budget.manna_spent += manna_spent
        if turns_spent:
            session.budget.turns_spent += turns_spent
