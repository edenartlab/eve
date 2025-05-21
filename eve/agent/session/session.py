from typing import Optional
from eve.agent.agent import Agent
from eve.agent.session.models import (
    ChatMessage,
    PromptSessionContext,
    Session,
    SessionUpdate,
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


def select_messages(session: Session, context: PromptSessionContext):
    messages = ChatMessage.get_collection()
    selected_messages = list(
        messages.find({"session": session.id}).sort("createdAt", -1).limit(10)
    )
    selected_messages.reverse()  # Reverse to get chronological order
    return selected_messages


async def build_llm_context(
    session: Session, actor: Agent, context: PromptSessionContext
):
    messages = select_messages(session, context)
    print(f"***debug*** messages: {messages}")
    new_message = ChatMessage(
        session=session.id, role="user", content=context.message.content
    )
    messages.append(new_message)
    print(f"***debug*** new_message: {new_message}")
    print(f"***debug*** messages: {messages}")
    tools = actor.get_tools(cache=False, auth_user=context.initiating_user_id)
    print(f"***debug*** tools: {tools}")
    return LLMContext(
        messages=messages,
        tools=tools,
        config=context.llm_config or LLMConfig(),
        metadata=LLMContextMetadata(
            trace_name="prompt_session",
            trace_id=str(f"prompt_session_{context.session.id}"),
            generation_name="prompt_session",
            generation_id=str(f"prompt_session_{context.session.id}"),
            trace_metadata={
                "session_id": str(context.session.id),
                "initiating_user_id": str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                "actor_agent_id": str(actor.id),
            },
        ),
    )


async def async_prompt_session(session: Session, llm_context: LLMContext):
    print("***debug*** entering async_prompt_session")
    print(f"***debug*** llm_context: {llm_context}")
    print("***debug*** yielding START_PROMPT")
    yield SessionUpdate(type=UpdateType.START_PROMPT)
    prompt_session_finished = False
    while not prompt_session_finished:
        print("***debug*** calling async_prompt")
        response = await async_prompt(llm_context)
        print(f"***debug*** response: {response}")
        assistant_message = ChatMessage(
            session=session.id,
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        )
        print(f"***debug*** response: {response}")
        llm_context.messages.append(assistant_message)
        print("***debug*** yielding ASSISTANT_MESSAGE")
        yield SessionUpdate(
            type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
        )

        if response.stop == "stop":
            prompt_session_finished = True

    print("***debug*** prompt session finished")
    yield SessionUpdate(type=UpdateType.END_PROMPT)


@handle_errors
async def run_prompt_session(context: PromptSessionContext):
    print("***debug*** running prompt session")
    print(f"***debug*** context: {context}")
    session = context.session
    validate_prompt_session(session, context)
    actor = await determine_actor(session, context)
    llm_context = await build_llm_context(session, actor, context)
    print(f"***debug*** llm_context: {llm_context}")
    print("***debug*** starting async_prompt_session")
    async for update in async_prompt_session(session, llm_context):
        print(f"***debug*** got update: {update}")
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

        print(f"***debug*** emitting update: {data}")
        await emit_update(context.update_config, data)
    print("***debug*** run_prompt_session finished")
