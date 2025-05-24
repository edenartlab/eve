import asyncio
import json
import traceback
from typing import List, Optional
import uuid

from bson import ObjectId
from sentry_sdk import capture_exception
from eve.agent.agent import Agent
from eve.agent.session.models import (
    ChatMessage,
    LLMTraceMetadata,
    PromptSessionContext,
    Session,
    SessionUpdate,
    ToolCall,
    UpdateType,
)
from eve.agent.session.session_llm import LLMContext, async_prompt
from eve.agent.session.models import LLMContextMetadata
from eve.api.errors import handle_errors
from eve.api.helpers import emit_update, serialize_for_json
from eve.agent.session.models import LLMConfig


def validate_prompt_session(session: Session, context: PromptSessionContext):
    if session.status == "archived":
        raise ValueError("Session is archived")


async def determine_actor(
    session: Session, context: PromptSessionContext
) -> Optional[Agent]:
    actor_id = None
    if context.actor_agent_id:
        actor_id = context.actor_agent_id
    elif len(session.agents) > 1:
        raise ValueError("Multi-agent smart sessions not yet implemented")
    elif len(session.agents) == 1:
        actor_id = session.agents[0]

    if not actor_id:
        # TODO: do something more graceful than returning None if no actor is determined to be necessary.
        return None

    actor = Agent.from_mongo(actor_id)
    return actor


def select_messages(
    session: Session, context: PromptSessionContext, selection_limit: int = 25
):
    messages = ChatMessage.get_collection()
    selected_messages = list(
        messages.find({"session": session.id})
        .sort("createdAt", -1)
        .limit(selection_limit)
    )
    selected_messages.reverse()
    selected_messages = [ChatMessage(**msg) for msg in selected_messages]
    return selected_messages


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Convert the role of any message that is not the actor to "user".

    This is experimentally how we are handling multi-agent sessions.
    """
    for message in messages:
        if message.sender != actor_id:
            message.role = "user"
    return messages


async def build_llm_context(
    session: Session, actor: Agent, context: PromptSessionContext
):
    tools = actor.get_tools(cache=False, auth_user=context.initiating_user_id)
    messages = select_messages(session, context)
    messages = convert_message_roles(messages, session.agents[0])
    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(context.initiating_user_id),
        role="user",
        content=context.message.content,
    )
    new_message.save()
    messages.append(new_message)
    return LLMContext(
        messages=messages,
        tools=tools,
        config=context.llm_config or LLMConfig(),
        metadata=LLMContextMetadata(
            # note - this is for observability purposes only. it is not the same as session.id
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
    llm_context: LLMContext,
    tool_call: ToolCall,
    tool_call_index: int,
):
    try:
        tool = llm_context.tools[tool_call.tool]
        task = await tool.async_start_task(
            user_id=llm_context.metadata.trace_metadata.user_id
            or llm_context.metadata.trace_metadata.agent_id,
            agent_id=llm_context.metadata.trace_metadata.agent_id,
            args=tool_call.args,
            mock=False,
            public=True,
            is_client_platform=False,
        )

        result = await tool.async_wait(task)
        tool_result_message = ChatMessage(
            session=ObjectId(llm_context.metadata.trace_metadata.session_id),
            sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
            name=tool_call.tool,
            tool_call_id=tool_call.id,
            role="tool",
            content=json.dumps(serialize_for_json(result)),
        )
        tool_result_message.save()
        llm_context.messages.append(tool_result_message)

        if result["status"] == "completed":
            return SessionUpdate(
                type=UpdateType.TOOL_COMPLETE,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result=result,
            )
        else:
            return SessionUpdate(
                type=UpdateType.ERROR,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                error=result.get("error"),
            )
    except Exception as e:
        capture_exception(e)
        traceback.print_exc()

        tool_result_message = ChatMessage(
            session=ObjectId(llm_context.metadata.trace_metadata.session_id),
            sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
            name=tool_call.tool,
            tool_call_id=tool_call.id,
            role="tool",
            content=serialize_for_json(e),
        )
        tool_result_message.save()
        llm_context.messages.append(tool_result_message)

        return SessionUpdate(
            type=UpdateType.ERROR,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            error=str(e),
        )


async def process_tool_calls(session: Session, llm_context: LLMContext):
    tool_calls = llm_context.messages[-1].tool_calls
    for b in range(0, len(tool_calls), 4):
        batch = enumerate(tool_calls[b : b + 4])
        tasks = [
            process_tool_call(
                session,
                llm_context,
                tool_call,
                b + idx,
            )
            for idx, tool_call in batch
        ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    for result in results:
        yield result


async def async_prompt_session(session: Session, llm_context: LLMContext):
    yield SessionUpdate(type=UpdateType.START_PROMPT)
    prompt_session_finished = False
    while not prompt_session_finished:
        response = await async_prompt(llm_context)
        assistant_message = ChatMessage(
            session=session.id,
            sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        )
        assistant_message.save()
        llm_context.messages.append(assistant_message)
        yield SessionUpdate(
            type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
        )

        if response.tool_calls:
            async for update in process_tool_calls(session, llm_context):
                yield update

        if response.stop == "stop":
            prompt_session_finished = True

    yield SessionUpdate(type=UpdateType.END_PROMPT)


@handle_errors
async def run_prompt_session(context: PromptSessionContext):
    session = context.session
    validate_prompt_session(session, context)
    actor = await determine_actor(session, context)
    llm_context = await build_llm_context(session, actor, context)
    async for update in async_prompt_session(session, llm_context):
        data = {
            "type": update.type.value,
            "update_config": context.update_config.model_dump()
            if context.update_config
            else None,
        }
        if update.type == UpdateType.START_PROMPT:
            pass
        elif update.type == UpdateType.ASSISTANT_MESSAGE:
            data["content"] = update.message.content
        elif update.type == UpdateType.TOOL_COMPLETE:
            data["tool"] = update.tool_name
            data["result"] = serialize_for_json(update.result)
        elif update.type == UpdateType.ERROR:
            data["error"] = update.error if hasattr(update, "error") else None
        elif update.type == UpdateType.END_PROMPT:
            pass

        await emit_update(context.update_config, data)
