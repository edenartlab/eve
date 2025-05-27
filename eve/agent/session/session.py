import asyncio
import json
import re
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
from eve.agent.session.session_llm import (
    LLMContext,
    async_prompt,
    async_prompt_stream,
    async_run_tool_call,
)
from eve.agent.session.models import LLMContextMetadata
from eve.api.errors import handle_errors
from eve.api.helpers import emit_update, serialize_for_json
from eve.agent.session.models import LLMConfig
from eve.agent.session.session_prompts import system_template


def validate_prompt_session(session: Session, context: PromptSessionContext):
    if session.status == "archived":
        raise ValueError("Session is archived")


def parse_mentions(content: str) -> List[str]:
    return re.findall(r"@(\w+)", content)


async def determine_actor(
    session: Session, context: PromptSessionContext
) -> Optional[Agent]:
    actor_id = None
    if context.actor_agent_id:
        actor_id = context.actor_agent_id
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


def build_system_message(session: Session, actor: Agent, context: PromptSessionContext):
    content = system_template.render(
        name=actor.name,
        persona=actor.persona,
        # TODO: add knowledge and models instructions
        knowledge="",
        models_instructions="",
    )
    return ChatMessage(
        session=session.id, sender=ObjectId(actor.id), role="system", content=content
    )


async def build_llm_context(
    session: Session, actor: Agent, context: PromptSessionContext
):
    system_message = build_system_message(session, actor, context)
    messages = [system_message]
    tools = actor.get_tools(cache=True, auth_user=context.initiating_user_id)
    messages.extend(select_messages(session, context))
    messages = convert_message_roles(messages, session.agents[0])
    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(context.initiating_user_id),
        role="user",
        content=context.message.content,
    )
    new_message.save()
    session.messages.append(new_message.id)
    session.save()
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
    # Update the tool call status to running
    tool_call.status = "running"

    # Find and update the original assistant message
    assistant_message = llm_context.messages[
        -1
    ]  # Last message should be the assistant message
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
        )
        print(f"***debug result: {result}")
        print(
            f"***debug result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
        )
        print(f"***debug result structure: {type(result)}")

        # Create tool result message for LLM context
        tool_result_message = ChatMessage(
            session=ObjectId(llm_context.metadata.trace_metadata.session_id),
            sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
            name=tool_call.tool,
            tool_call_id=tool_call.id,
            role="tool",
            content=json.dumps(serialize_for_json(result)),
        )
        tool_result_message.save()
        session.messages.append(tool_result_message.id)
        session.save()
        llm_context.messages.append(tool_result_message)

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
                assistant_message.save()

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

        # Create tool result message for LLM context
        tool_result_message = ChatMessage(
            session=ObjectId(llm_context.metadata.trace_metadata.session_id),
            sender=ObjectId(llm_context.metadata.trace_metadata.agent_id),
            name=tool_call.tool,
            tool_call_id=tool_call.id,
            role="tool",
            content=serialize_for_json(e),
        )
        tool_result_message.save()
        session.messages.append(tool_result_message.id)
        session.save()
        llm_context.messages.append(tool_result_message)

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


async def async_prompt_session(
    session: Session, llm_context: LLMContext, stream: bool = False
):
    yield SessionUpdate(type=UpdateType.START_PROMPT)
    prompt_session_finished = False
    while not prompt_session_finished:
        if stream:
            # For streaming, we need to collect the content as it comes in
            content = ""
            tool_calls_dict = {}  # Track tool calls by index to accumulate arguments
            stop_reason = None

            async for chunk in async_prompt_stream(llm_context):
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        content += choice.delta.content
                        yield SessionUpdate(
                            type=UpdateType.ASSISTANT_TOKEN, text=choice.delta.content
                        )
                    if choice.delta and choice.delta.tool_calls:
                        # Handle tool calls in streaming - accumulate arguments
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
                        print(
                            f"***debug failed to parse tool call arguments: {tc_data['arguments']}"
                        )
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

        assistant_message.save()
        session.messages.append(assistant_message.id)
        session.save()
        llm_context.messages.append(assistant_message)
        yield SessionUpdate(
            type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
        )

        if assistant_message.tool_calls:
            async for update in process_tool_calls(session, llm_context):
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
        pass
    elif update.type == UpdateType.ASSISTANT_TOKEN:
        data["text"] = update.text
    elif update.type == UpdateType.ASSISTANT_MESSAGE:
        data["content"] = update.message.content
        if update.message.tool_calls:
            data["tool_calls"] = [
                serialize_for_json(tc.model_dump()) for tc in update.message.tool_calls
            ]
    elif update.type == UpdateType.TOOL_COMPLETE:
        data["tool"] = update.tool_name
        data["result"] = serialize_for_json(update.result)
    elif update.type == UpdateType.ERROR:
        data["error"] = update.error if hasattr(update, "error") else None
    elif update.type == UpdateType.END_PROMPT:
        pass

    return data


async def _run_prompt_session_internal(
    context: PromptSessionContext, stream: bool = False
):
    """Internal function that handles both streaming and non-streaming"""
    session = context.session
    validate_prompt_session(session, context)
    actor = await determine_actor(session, context)
    llm_context = await build_llm_context(session, actor, context)

    async for update in async_prompt_session(session, llm_context, stream=stream):
        yield format_session_update(update, context)


async def run_prompt_session_stream(context: PromptSessionContext):
    try:
        async for data in _run_prompt_session_internal(context, stream=True):
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
async def run_prompt_session(context: PromptSessionContext):
    async for data in _run_prompt_session_internal(context, stream=False):
        await emit_update(context.update_config, data)
