from typing import Optional
from eve.agent.agent import Agent
from eve.agent.session.models import ChatMessage, PromptSessionContext, Session
from eve.agent.session.session_llm import LLMContext, async_prompt
from eve.agent.session.models import LLMContextMetadata


def validate_prompt_session(session: Session, context: PromptSessionContext):
    if session.status == "archived":
        raise ValueError("Session is archived")


async def determine_actor(
    session: Session, context: PromptSessionContext
) -> Optional[Agent]:
    actor_id = None
    if len(session.agents) > 1:
        raise ValueError("Multi-agent sessions not yet implemented")
    else:
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


async def prompt_session(context: PromptSessionContext):
    session = Session.from_mongo(context.session_id)
    validate_prompt_session(session, context)
    actor = await determine_actor(session, context)
    llm_context = await build_llm_context(session, actor, context)
    await async_prompt(llm_context)
    # now we do the prompt thread stuff