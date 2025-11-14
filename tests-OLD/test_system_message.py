import pytest

from eve.agent.agent import Agent
from eve.agent.session.models import Session
from eve.agent.session.session import build_system_message
from eve.agent.memory.service import memory_service
from eve.user import User


@pytest.mark.asyncio
async def test_memory_context():
    session = Session.from_mongo("68cc2bc9d012fbd52b1a5dad")
    agent = Agent.load("eden_gigabrain")
    user = User.from_mongo(session.last_actor_id)

    memory_context = await memory_service.assemble_memory_context(session, agent, user)

    assert memory_context is not None


@pytest.mark.asyncio
async def test_system_message():
    session = Session.from_mongo("68cc2bc9d012fbd52b1a5dad")
    agent = Agent.load("eden_gigabrain")
    user = User.from_mongo(session.last_actor_id)
    tools = agent.get_tools(auth_user=user.id)

    system_message = await build_system_message(
        session,
        agent,
        user,
        tools,
    )

    assert system_message is not None
    assert hasattr(system_message, "content")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_system_message())
    asyncio.run(test_memory_context())
