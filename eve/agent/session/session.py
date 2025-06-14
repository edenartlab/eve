import asyncio
import json
import os
import random
import re
import traceback
from fastapi import BackgroundTasks
import pytz
from typing import List, Optional
import uuid
from datetime import datetime
from bson import ObjectId
from sentry_sdk import capture_exception
from eve.eden_utils import dumps_json, dumps_json
from eve.agent.agent import Agent
from eve.agent.session.models import (
    ActorSelectionMethod,
    ChatMessage,
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
from eve.agent.session.session_prompts import system_template


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


def parse_mentions(content: str) -> List[str]:
    return re.findall(r"@(\w+)", content)


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
    session: Session, context: PromptSessionContext
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
    elif len(session.agents) > 1:
        mentions = parse_mentions(context.message.content)
        if len(mentions) > 1:
            raise ValueError("Multiple @mentions not currently supported")
        elif len(mentions) == 1:
            mentioned_username = mentions[0]
            for agent_id in session.agents:
                agent = Agent.from_mongo(agent_id)
                if agent.username == mentioned_username:
                    actor_id = agent_id
                    break
            if not actor_id:
                raise ValueError(f"Agent @{mentioned_username} not found in session")
    elif len(session.agents) == 1:
        actor_id = session.agents[0]

    if not actor_id:
        # TODO: do something more graceful than returning None if no actor is determined to be necessary.
        return None

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
    return selected_messages


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Re-assembles messages from perspective of actor (assistant) and everyone else (user)
    """
    messages = [
        message.as_assistant_message()
        if message.sender == actor_id
        else message.as_user_message()
        for message in messages
    ]
    return messages


def build_system_message(session: Session, actor: Agent, context: PromptSessionContext):
    content = system_template.render(
        name=actor.name,
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        persona=actor.persona,
        scenario=session.scenario,
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
            session_id=str(uuid.uuid4()),
            trace_name="prompt_session",
            trace_id=str(f"prompt_session_{context.session.id}"),
            generation_name="prompt_session",
            generation_id=str(f"prompt_session_{context.session.id}"),
            trace_metadata=LLMTraceMetadata(
                user_id=str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                agent_id=str(actor.id),
                session_id=str(context.session.id),
            ),
        ),
    )


async def process_tool_call(
    session: Session,
    assistant_message: ChatMessage,
    llm_context: LLMContext,
    tool_call: ToolCall,
    tool_call_index: int,
):
    # Update the tool call status to running
    tool_call.status = "running"

    # Update assistant message
    if assistant_message.tool_calls and tool_call_index < len(
        assistant_message.tool_calls
    ):
        assistant_message.tool_calls[tool_call_index].status = "running"
        assistant_message.save()

    try:
        result = await async_run_tool_call(
            llm_context,
            tool_call,
            user_id=llm_context.metadata.trace_metadata.user_id
            or llm_context.metadata.trace_metadata.agent_id,
            agent_id=llm_context.metadata.trace_metadata.agent_id,
            session_id=llm_context.metadata.trace_metadata.session_id,
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
    session: Session, assistant_message: ChatMessage, llm_context: LLMContext
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
            )
            for idx, tool_call in batch
        ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    for result in results:
        yield result


async def async_prompt_session(
    session: Session, llm_context: LLMContext, actor: Agent, stream: bool = False
):
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
        if stream:
            # For streaming, we need to collect the content as it comes in
            content = ""
            tool_calls_dict = {}  # Track tool calls by index to accumulate arguments
            stop_reason = None
            tokens_spent = 0  # Initialize tokens_spent for streaming

            async for chunk in async_prompt_stream(llm_context):
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    # Only yield content tokens, not tool call chunks
                    if choice.delta and choice.delta.content:
                        content += choice.delta.content
                        yield SessionUpdate(
                            type=UpdateType.ASSISTANT_TOKEN, text=choice.delta.content
                        )
                    # Process tool calls silently (don't yield anything)
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
                session, assistant_message, llm_context
            ):
                yield update

        if stop_reason == "stop":
            prompt_session_finished = True

    yield SessionUpdate(type=UpdateType.END_PROMPT)


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
    actor = await determine_actor(session, context)
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
