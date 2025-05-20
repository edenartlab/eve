from typing import Optional, List
from bson import ObjectId
from eve.agent.session.models import ChatMessage, PromptSessionContext, Session
from eve.agent.session.session_llm import LLMConfig, LLMContext, async_prompt


def create_session(
    owner: ObjectId,
    title: str,
    agents: List[ObjectId],
    scenario: Optional[str] = None,
    budget: Optional[float] = None,
):
    session = Session(
        owner=owner, title=title, agents=agents, scenario=scenario, budget=budget
    )
    session.save()
    return session


def list_sessions(owner: ObjectId):
    return Session.find_many(Session.owner == owner)


def get_session(session_id: ObjectId):
    return Session.from_mongo(session_id)


def archive_session(session_id: ObjectId):
    session = Session.from_mongo(session_id)
    session.status = "archived"
    session.save()
    return session


async def prompt_session(
    context: PromptSessionContext, config: Optional[LLMConfig] = None
):
    # TODO: Determine who should act, what messages are available in context, and what tools are available
    messages = [ChatMessage(role="user", content=context.message)]
    tools = []
    context = LLMContext(
        session_id=context.session_id,
        initiating_user_id=context.initiating_user_id,
        messages=messages,
        tools=tools,
    )
    await async_prompt(context, config)
