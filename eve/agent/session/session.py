import asyncio
import json
import os
import random
import re
import traceback
from fastapi import BackgroundTasks
import pytz
from typing import List, Optional, Dict
import uuid
from datetime import datetime
from bson import ObjectId
from sentry_sdk import capture_exception
from eve.eden_utils import dumps_json
from eve.agent.agent import Agent
from eve.tool import Tool
from eve.mongo import get_collection
from eve.models import Model
from eve.agent.session.models import (
    ActorSelectionMethod,
    ChatMessage,
    ChatMessageObservability,
    LLMTraceMetadata,
    PromptSessionContext,
    Session,
    SessionUpdate,
    ToolCall,
    UpdateType,
)
from eve.agent.session.session_llm import (
    LLMContext,
    async_prompt,
    async_prompt_stream,
    async_run_tool_call,
)
from eve.agent.session.models import LLMContextMetadata
from eve.api.errors import handle_errors
from eve.api.helpers import emit_update
from eve.agent.session.models import LLMConfig
from eve.agent.session.session_prompts import (
    system_template,
    model_template,
)


class SessionCancelledException(Exception):
    """Exception raised when a session is cancelled via Ably signal."""

    pass


def check_session_budget(session: Session):
    if session.budget:
        if session.budget.token_budget:
            if session.budget.tokens_spent >= session.budget.token_budget:
                raise ValueError("Session token budget exceeded")
        if session.budget.manna_budget:
            if session.budget.manna_spent >= session.budget.manna_budget:
                raise ValueError("Session manna budget exceeded")
        if session.budget.turn_budget:
            if session.budget.turns_spent >= session.budget.turn_budget:
                raise ValueError("Session turn budget exceeded")


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


def validate_prompt_session(session: Session, context: PromptSessionContext):
    if session.status == "archived":
        raise ValueError("Session is archived")
    has_budget = session.budget and (
        session.budget.token_budget
        or session.budget.manna_budget
        or session.budget.turn_budget
    )
    if (
        session.autonomy_settings
        and session.autonomy_settings.auto_reply
        and not has_budget
    ):
        raise ValueError("Session cannot have auto-reply enabled without a set budget")
    if has_budget:
        check_session_budget(session)


def determine_actor_from_actor_selection_method(session: Session) -> Optional[Agent]:
    selection_method = session.autonomy_settings.actor_selection_method
    if (
        session.autonomy_settings.actor_selection_method
        == ActorSelectionMethod.RANDOM_EXCLUDE_LAST
    ):
        if not session.last_actor_id:
            return random.choice(session.agents)
        last_actor_id = session.last_actor_id
        eligible_actors = [
            agent_id for agent_id in session.agents if agent_id != last_actor_id
        ]
        return random.choice(eligible_actors)
    elif selection_method == ActorSelectionMethod.RANDOM:
        return random.choice(session.agents)
    else:
        raise ValueError(f"Invalid actor selection method: {selection_method}")


async def determine_actor(
    session: Session, context: PromptSessionContext, agents: List[Agent]
) -> Optional[Agent]:
    actor_id = None
    if context.actor_agent_id:
        # Default to the actor specified in the context if passed
        requested_actor = ObjectId(context.actor_agent_id)
        if requested_actor in session.agents:
            actor_id = context.actor_agent_id
        else:
            raise ValueError(f"Actor @{context.actor_agent_id} not found in session")
    elif session.autonomy_settings and session.autonomy_settings.auto_reply:
        actor_id = determine_actor_from_actor_selection_method(session)
    elif len(session.agents) == 1:
        actor_id = session.agents[0]
    elif len(session.agents) > 1:
        mentions, force_mentions = [], []
        for agent in agents:
            # Force mentions: @name or @username
            force_pattern = rf"\b@(?:{re.escape(agent.username)}|{re.escape(agent.name)})\b"
            if re.search(force_pattern, context.message.content, re.IGNORECASE):
                force_mentions.append(agent)
            
            # Regular mentions: name or username as whole words
            regular_pattern = rf"\b(?:{re.escape(agent.username)}|{re.escape(agent.name)})\b"
            if re.search(regular_pattern, context.message.content, re.IGNORECASE):
                mentions.append(agent)
        if len(force_mentions) == 1:
            actor_id = force_mentions[0].id
        elif len(force_mentions) > 1:
            actor_id = random.choice(force_mentions).id
        elif len(mentions) == 1:
            actor_id = mentions[0].id
        elif len(mentions) > 1:
            actor_id = random.choice(mentions).id


    if not actor_id:
        # TODO: governor/dispatcher here
        actor_id = random.choice(session.agents)

    actor = Agent.from_mongo(actor_id)
    session.last_actor_id = actor.id
    session.save()
    return actor


def select_messages(session: Session, selection_limit: int = 25):
    messages = ChatMessage.get_collection()
    selected_messages = list(
        messages.find({"session": session.id, "role": {"$ne": "eden"}})
        .sort("createdAt", -1)
        .limit(selection_limit)
    )
    selected_messages.reverse()
    selected_messages = [ChatMessage(**msg) for msg in selected_messages]
    # Filter out cancelled tool calls from the messages
    selected_messages = [msg.filter_cancelled_tool_calls() for msg in selected_messages]
    return selected_messages


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Re-assembles messages from perspective of actor (assistant) and everyone else (user)
    """
    
    for message in messages:
        print("sender", message.sender, "role", message.role)

    messages = [
        message.as_assistant_message()
        if message.sender == actor_id
        else message.as_user_message()
        for message in messages
    ]

    return messages


def build_system_message(session: Session, actor: Agent, context: PromptSessionContext):
    # Get text describing models
    if actor.models:
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in actor.models}
        lora_docs = models_collection.find(
            {"_id": {"$in": list(loras_dict.keys())}, "deleted": {"$ne": True}}
        )
        lora_docs = list(lora_docs or [])
        for doc in lora_docs:
            doc["use_when"] = loras_dict[ObjectId(doc["_id"])].get("use_when", "This is your default Lora model")
        loras = "\n".join(model_template.render(doc) for doc in lora_docs or [])
    else:
        loras = ""

    content = system_template.render(
        name=actor.name,
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        persona=actor.persona,
        scenario=session.scenario,
        loras=loras,
        voice=actor.voice,
    )

    return ChatMessage(
        session=session.id, sender=ObjectId(actor.id), role="system", content=content
    )


def add_user_message(session: Session, context: PromptSessionContext):
    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(context.initiating_user_id),
        role="user",
        content=context.message.content,
    )
    new_message.save()
    session.messages.append(new_message.id)
    session.save()
    return new_message


async def build_llm_context(
    session: Session, actor: Agent, context: PromptSessionContext
):
    tools = actor.get_tools(cache=True, auth_user=context.initiating_user_id)
    
    # pass custom tools
    if context.custom_tools:
        tools.update(context.custom_tools)
        tools = {tool: tools[tool] for tool in tools if tool in context.custom_tools}

    # set voice default if tools include elevenlabs
    if actor.voice and "elevenlabs" in tools.keys():
        tools["elevenlabs"] = Tool.from_raw_yaml({
            "parent_tool": "elevenlabs", 
            "parameters": {"voice": {"default": actor.voice}}}
        )

    # if models
    if actor.models:
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in actor.models}
        lora_docs = models_collection.find(
            {"_id": {"$in": list(loras_dict.keys())}, "deleted": {"$ne": True}}
        )
        lora_docs = list(lora_docs or [])

    # build messages
    system_message = build_system_message(session, actor, context)
    messages = [system_message]
    messages.extend(select_messages(session))
    messages = convert_message_roles(messages, actor.id)
    
    if context.initiating_user_id:
        new_message = add_user_message(session, context)
        messages.append(new_message)

    return LLMContext(
        messages=messages,
        tools=tools,
        config=context.llm_config or LLMConfig(),
        metadata=LLMContextMetadata(
            # for observability purposes. not same as session.id
            session_id=f"{os.getenv('DB')}-{str(context.session.id)}",
            trace_name="prompt_session",
            trace_id=str(uuid.uuid4()),
            generation_name="prompt_session",
            trace_metadata=LLMTraceMetadata(
                user_id=str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                agent_id=str(actor.id),
                session_id=str(context.session.id),
            ),
        ),
    )


async def async_run_tool_call_with_cancellation(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
    cancellation_event: asyncio.Event = None,
):
    """
    Cancellation-aware version of async_run_tool_call that can be interrupted
    """
    tool = llm_context.tools[tool_call.tool]

    # Start the task
    task = await tool.async_start_task(
        user_id=user_id,
        agent_id=agent_id,
        args=tool_call.args,
        mock=False,
        public=public,
        is_client_platform=is_client_platform,
    )

    # If no cancellation event, fall back to normal behavior
    if not cancellation_event:
        result = await tool.async_wait(task)
    else:
        # Race between task completion and cancellation
        wait_task = asyncio.create_task(tool.async_wait(task))

        try:
            # Wait for either task completion or cancellation
            done, pending = await asyncio.wait(
                [wait_task, asyncio.create_task(cancellation_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks
            for task_obj in pending:
                task_obj.cancel()
                try:
                    await task_obj
                except asyncio.CancelledError:
                    pass

            # Check if cancellation happened first
            if cancellation_event.is_set():
                # Try to cancel the task
                try:
                    await tool.async_cancel(task)
                except Exception as e:
                    print(f"Failed to cancel task {task.id}: {e}")

                return {"status": "cancelled", "error": "Task cancelled by user"}
            else:
                # Task completed normally
                result = wait_task.result()

        except Exception as e:
            # If anything goes wrong, try to cancel the task
            try:
                if not wait_task.done():
                    wait_task.cancel()
                    await tool.async_cancel(task)
            except:
                pass
            raise e

    # Add task.cost and task.id to the result object
    if isinstance(result, dict):
        result["cost"] = getattr(task, "cost", None)
        result["task"] = getattr(task, "id", None)

    return result


async def process_tool_call(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    tool_call: ToolCall,
    tool_call_index: int,
    cancellation_event: asyncio.Event = None,
):
    # Update the tool call status to running
    tool_call.status = "running"

    # Update assistant message
    if assistant_message.tool_calls and tool_call_index < len(
        assistant_message.tool_calls
    ):
        assistant_message.tool_calls[tool_call_index].status = "running"
        try:
            # Force save with direct MongoDB update
            from eve.mongo import get_collection

            messages_collection = get_collection("messages")
            result = messages_collection.update_one(
                {"_id": assistant_message.id},
                {"$set": {f"tool_calls.{tool_call_index}.status": "running"}},
            )
        except Exception as e:
            print(f"Failed to update tool status to running: {e}")
            # Try regular save as fallback
            assistant_message.save()

    try:
        # Check for cancellation before starting tool execution
        if cancellation_event and cancellation_event.is_set():
            tool_call.status = "cancelled"
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.tool_calls[tool_call_index].status = "cancelled"
                try:
                    # Force save with direct MongoDB update
                    from eve.mongo import get_collection

                    messages_collection = get_collection("messages")
                    result = messages_collection.update_one(
                        {"_id": assistant_message.id},
                        {"$set": {f"tool_calls.{tool_call_index}.status": "cancelled"}},
                    )
                    print(
                        f"Direct MongoDB update for tool {tool_call_index}: {result.modified_count} modified"
                    )
                except Exception as e:
                    assistant_message.save()
            return SessionUpdate(
                type=UpdateType.TOOL_CANCELLED,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result={"status": "cancelled"},
            )

        # Use cancellation-aware tool execution
        result = await async_run_tool_call_with_cancellation(
            llm_context,
            tool_call,
            user_id=llm_context.metadata.trace_metadata.user_id
            or llm_context.metadata.trace_metadata.agent_id,
            agent_id=llm_context.metadata.trace_metadata.agent_id,
            cancellation_event=cancellation_event,
        )

        # Check for cancellation after tool execution completes
        if cancellation_event and cancellation_event.is_set():
            tool_call.status = "cancelled"
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.tool_calls[tool_call_index].status = "cancelled"
                try:
                    # Force save with direct MongoDB update
                    from eve.mongo import get_collection

                    messages_collection = get_collection("messages")
                    result = messages_collection.update_one(
                        {"_id": assistant_message.id},
                        {"$set": {f"tool_calls.{tool_call_index}.status": "cancelled"}},
                    )
                except Exception as e:
                    print(f"Direct update failed, trying regular save: {e}")
                    assistant_message.save()
            return SessionUpdate(
                type=UpdateType.TOOL_CANCELLED,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result={"status": "cancelled"},
            )

        # Update the original tool call with result
        if result["status"] == "completed":
            tool_call.status = "completed"
            # Extract the actual tool result based on the known structure
            # result["result"] contains the list of result objects
            tool_result = result.get("result", [])

            tool_call.result = tool_result
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.tool_calls[tool_call_index].status = "completed"
                assistant_message.tool_calls[tool_call_index].result = tool_result
                assistant_message.tool_calls[tool_call_index].cost = result.get(
                    "cost", 0
                )
                assistant_message.tool_calls[tool_call_index].task = result.get("task")
                assistant_message.observability = ChatMessageObservability(
                    session_id=llm_context.metadata.session_id,
                    trace_id=llm_context.metadata.trace_id,
                    tokens_spent=0,
                )
                assistant_message.save()

            update_session_budget(session, manna_spent=result.get("cost", 0))
            session.save()

            return SessionUpdate(
                type=UpdateType.TOOL_COMPLETE,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result=result,
            )
        else:
            tool_call.status = "failed"
            tool_call.error = result.get("error")
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.tool_calls[tool_call_index].status = "failed"
                assistant_message.tool_calls[tool_call_index].error = result.get(
                    "error"
                )
                assistant_message.save()

            return SessionUpdate(
                type=UpdateType.ERROR,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                error=result.get("error"),
            )
    except Exception as e:
        capture_exception(e)
        traceback.print_exc()

        # Update the original tool call with error
        tool_call.status = "failed"
        tool_call.error = str(e)
        if assistant_message.tool_calls and tool_call_index < len(
            assistant_message.tool_calls
        ):
            assistant_message.tool_calls[tool_call_index].status = "failed"
            assistant_message.tool_calls[tool_call_index].error = str(e)
            assistant_message.save()

        return SessionUpdate(
            type=UpdateType.ERROR,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            error=str(e),
        )


async def process_tool_calls(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    cancellation_event: asyncio.Event = None,
):
    tool_calls = assistant_message.tool_calls
    for b in range(0, len(tool_calls), 4):
        batch = enumerate(tool_calls[b : b + 4])
        tasks = [
            process_tool_call(
                session,
                assistant_message,
                llm_context,
                tool_call,
                b + idx,
                cancellation_event,
            )
            for idx, tool_call in batch
        ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    for result in results:
        yield result


async def async_prompt_session(
    session: Session, llm_context: LLMContext, actor: Agent, stream: bool = False
):
    # Set up cancellation handling via Ably
    cancellation_event = asyncio.Event()
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
                    cancellation_event.set()
            except Exception as e:
                print(f"Error in cancellation handler: {e}")

        await channel.subscribe("cancel", cancellation_handler)

    except Exception as e:
        print(f"Failed to setup Ably cancellation for session {session.id}: {e}")
        # Continue without cancellation support if Ably fails

    async def prompt_session_generator():
        """Generator function that yields session updates and can be cancelled."""
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
        )

        prompt_session_finished = False
        tokens_spent = 0

        while not prompt_session_finished:
            # Check for cancellation before each iteration
            if cancellation_event.is_set():
                raise SessionCancelledException("Session cancelled by user")

            if stream:
                # For streaming, we need to collect the content as it comes in
                content = ""
                tool_calls_dict = {}  # Track tool calls by index to accumulate arguments
                stop_reason = None
                tokens_spent = 0  # Initialize tokens_spent for streaming

                async for chunk in async_prompt_stream(llm_context):
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
                    observability=ChatMessageObservability(
                        session_id=llm_context.metadata.session_id,
                        trace_id=llm_context.metadata.trace_id,
                        tokens_spent=tokens_spent,
                    ),
                )
            else:
                # Non-streaming path
                response = await async_prompt(llm_context)
                assistant_message = ChatMessage(
                    session=session.id,
                    sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    observability=ChatMessageObservability(
                        session_id=llm_context.metadata.session_id,
                        trace_id=llm_context.metadata.trace_id,
                        tokens_spent=response.tokens_spent,
                    ),
                )
                stop_reason = response.stop
                tokens_spent = response.tokens_spent

            assistant_message.save()
            session.messages.append(assistant_message.id)
            update_session_budget(session, tokens_spent=tokens_spent, turns_spent=1)
            session.save()
            llm_context.messages.append(assistant_message)
            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
            )

            if assistant_message.tool_calls:
                async for update in process_tool_calls(
                    session, assistant_message, llm_context, cancellation_event
                ):
                    # Check for cancellation during tool execution
                    if cancellation_event.is_set():
                        raise SessionCancelledException("Session cancelled by user")
                    yield update

            if stop_reason == "stop":
                prompt_session_finished = True

        yield SessionUpdate(type=UpdateType.END_PROMPT)

    try:
        # Run the prompt session generator, checking for cancellation
        async for update in prompt_session_generator():
            yield update

    except SessionCancelledException:
        # Handle graceful cancellation
        try:
            # 1. Mark any unfinished tool calls as cancelled
            last_message = None
            if session.messages:
                last_message_id = session.messages[-1]
                last_message = ChatMessage.from_mongo(last_message_id)

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
                            )

                    # Force save by updating the entire tool_calls array
                    try:
                        # Save using direct MongoDB update to ensure the change persists
                        from eve.mongo import get_collection

                        messages_collection = get_collection("messages")
                        result = messages_collection.update_one(
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
                    except Exception as e:
                        last_message.markModified("tool_calls")
                        last_message.save()

            # 2. Add system message indicating cancellation
            cancel_message = ChatMessage(
                session=session.id,
                sender=ObjectId("000000000000000000000000"),  # System sender
                role="system",
                content="Response cancelled by user",
            )
            cancel_message.save()
            session.messages.append(cancel_message.id)
            session.save()

            # 3. Yield final updates
            yield SessionUpdate(
                type=UpdateType.ASSISTANT_MESSAGE, message=cancel_message
            )
            yield SessionUpdate(type=UpdateType.END_PROMPT)

        except Exception as e:
            print(f"Error during session cancellation cleanup: {e}")
            yield SessionUpdate(type=UpdateType.END_PROMPT)

    finally:
        # Clean up Ably subscription
        if ably_client:
            try:
                await ably_client.close()
            except Exception as e:
                print(f"Error closing Ably client: {e}")


def format_session_update(update: SessionUpdate, context: PromptSessionContext) -> dict:
    """Convert SessionUpdate to the format expected by handlers"""
    data = {
        "type": update.type.value,
        "update_config": context.update_config.model_dump()
        if context.update_config
        else None,
    }

    if update.type == UpdateType.START_PROMPT:
        if update.agent:
            data["agent"] = update.agent
        # Include session_id in start_prompt event for frontend to capture
        data["session_id"] = str(context.session.id)
    elif update.type == UpdateType.ASSISTANT_TOKEN:
        data["text"] = update.text
    elif update.type == UpdateType.ASSISTANT_MESSAGE:
        data["content"] = update.message.content
        if update.message.tool_calls:
            data["tool_calls"] = [
                dumps_json(tc.model_dump()) for tc in update.message.tool_calls
            ]
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
    validate_prompt_session(session, context)
    agents = [Agent.from_mongo(agent_id) for agent_id in session.agents]
    actor = await determine_actor(session, context, agents)
    llm_context = await build_llm_context(session, actor, context)

    async for update in async_prompt_session(
        session, llm_context, actor, stream=stream
    ):
        yield format_session_update(update, context)

    if session.autonomy_settings and session.autonomy_settings.auto_reply:
        background_tasks.add_task(
            _queue_session_action_fastify_background_task, session
        )


async def run_prompt_session_stream(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    try:
        async for data in _run_prompt_session_internal(
            context, background_tasks, stream=True
        ):
            yield data
    except Exception as e:
        traceback.print_exc()
        yield {
            "type": UpdateType.ERROR.value,
            "error": str(e),
            "update_config": context.update_config.model_dump()
            if context.update_config
            else None,
        }


@handle_errors
async def run_prompt_session(
    context: PromptSessionContext, background_tasks: BackgroundTasks
):
    async for data in _run_prompt_session_internal(
        context, background_tasks, stream=False
    ):
        await emit_update(context.update_config, data)


async def _queue_session_action_fastify_background_task(session: Session):
    import httpx

    if session.autonomy_settings:
        await asyncio.sleep(session.autonomy_settings.reply_interval)

    url = f"{os.getenv('EDEN_API_URL')}/sessions/prompt"
    payload = {"session_id": str(session.id)}
    headers = {"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except Exception as e:
        print("HTTP request failed:", str(e))
        capture_exception(e)


async def async_title_session(
    session: Session, initial_message_content: str, metadata: Optional[Dict] = None
):
    """
    Generate a title for a session based on the initial message content
    """

    from pydantic import BaseModel, Field

    class TitleResponse(BaseModel):
        """A title for a session of chat messages. It must entice a user to click on the session when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat session. It should be concise and terse, and not include any special characters or punctuation."
        )

    try:
        if not initial_message_content:
            # If no message content, return without setting a title
            return

        # Add a system message and the initial user message for title generation
        system_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="system",
            content="You are an expert at creating concise titles for chat sessions.",
        )

        # Add the initial user message
        user_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender (placeholder)
            role="user",
            content=initial_message_content,
        )

        # Add request message for title generation
        request_message = ChatMessage(
            session=session.id,
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="user",
            content="Come up with a title for this session based on the user's message.",
        )

        # Build message list
        messages = [system_message, user_message, request_message]

        # Create LLM context
        llm_context = LLMContext(
            messages=messages,
            tools={},  # No tools needed for title generation
            config=LLMConfig(model="gpt-4o-mini", response_format=TitleResponse),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-{str(session.id)}",
                trace_name="FN_title_session",
                trace_id=str(uuid.uuid4()),
                generation_name="FN_title_session",
                trace_metadata=LLMTraceMetadata(
                    session_id=str(session.id),
                ),
            ),
        )

        # Generate title using async_prompt
        result = await async_prompt(llm_context)

        # Parse the response
        if hasattr(result, "content") and result.content:
            try:
                # Try to parse as JSON if response_format was used
                import json

                title_data = json.loads(result.content)
                if isinstance(title_data, dict) and "title" in title_data:
                    session.title = title_data["title"]
                else:
                    # Fallback to using content directly
                    session.title = result.content[:30]  # Limit to 30 chars
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, use content directly
                session.title = result.content[:30]  # Limit to 30 chars

            session.save()

    except Exception as e:
        capture_exception(e)
        traceback.print_exc()
        return


def title_session(
    session: Session, initial_message_content: str, metadata: Optional[Dict] = None
):
    """Synchronous wrapper for async_title_session"""
    return asyncio.run(async_title_session(session, initial_message_content))
