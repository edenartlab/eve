from typing import Optional
from eve.agent.agent import Agent
from eve.agent.session.models import (
    ChatMessage,
    PromptSessionContext,
    Session,
    UpdateType,
)
from eve.agent.session.session_llm import LLMContext, async_prompt
from eve.agent.session.models import LLMContextMetadata
from eve.api.helpers import emit_update, serialize_for_json


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


async def build_llm_context(
    session: Session, actor: Agent, context: PromptSessionContext
):
    # Get last 10 messages efficiently using MongoDB's sort and limit
    messages = ChatMessage.get_collection()
    selected_messages = (
        messages.load(session=session.id).sort("createdAt", -1).limit(10)
    )
    messages.reverse()  # Reverse to get chronological order

    # Add new message
    selected_messages.append(ChatMessage(role="user", content=context.message))

    return LLMContext(
        messages=messages,
        tools=actor.tools,
        metadata=LLMContextMetadata(
            trace_name="prompt_session",
            trace_id=str(f"prompt_session_{context.session_id}"),
            generation_name="prompt_session",
            generation_id=str(f"prompt_session_{context.session_id}"),
            trace_metadata={
                "session_id": str(context.session_id),
                "initiating_user_id": str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                "actor_agent_id": str(actor.id),
            },
        ),
    )


async def async_prompt_session(llm_context: LLMContext):
    async for update in async_prompt(llm_context):
        print(update)


async def run_prompt_session(context: PromptSessionContext):
    session = Session.from_mongo(context.session_id)
    validate_prompt_session(session, context)
    actor = await determine_actor(session, context)
    llm_context = await build_llm_context(session, actor, context)
    async for update in async_prompt_session(llm_context):
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
