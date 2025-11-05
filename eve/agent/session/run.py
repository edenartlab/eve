import asyncio
import json
import pytz
import logging
import uuid
from bson import ObjectId
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime

from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.user import User
from eve.tool import Tool
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import (
    system_template,
    conductor_template
)
from eve.agent.session.models import (
    Session,
    ChatMessage,
    PromptSessionContext,
    LLMConfig,
    LLMContext
)
from eve.agent.session.session import (
    add_chat_message,
    build_llm_context,
    async_prompt_session,
)

logger = logging.getLogger(__name__)


async def remote_prompt_session(
    session_id: str,
    agent_id: str,
    user_id: str,
    content: str,
    attachments: Optional[List[str]] = [],
    extra_tools: Optional[List[str]] = [],
):
    logger.info(
        f"Remote prompt: session={session_id}, agent={agent_id}, user={user_id}"
    )

    # Load models
    session = Session.from_mongo(session_id)
    agent = Agent.from_mongo(agent_id)
    user = User.from_mongo(user_id)

    # Create user message
    new_message = ChatMessage(
        role="user",
        sender=user.id,
        session=session.id,
        content=content,
        attachments=attachments,
    )

    # Build context
    context = PromptSessionContext(
        session=session,
        initiating_user_id=str(user.id),
        message=new_message,
        llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
    )

    if extra_tools:
        context.extra_tools = {k: Tool.load(k) for k in extra_tools}

    # Add message to session
    await add_chat_message(session, context)

    # Build LLM context and prompt
    context = await build_llm_context(
        session,
        agent,
        context,
        trace_id=str(uuid.uuid4()),
    )

    # Run the prompt
    async for m in async_prompt_session(session, context, agent):
        pass

    logger.info(f"Remote prompt completed for session {session_id}")








async def conductor():
    available_agents = ["kweku", "shuijing", "mycos"]
    session_id = "69067bdac9ab119e414bfff1"
    scenario = "Kweku, Shuijing and Mycos are debating the nature of consciousness. Kweku thinks consciousness is a software program, while the others disagree and gang up on him. Make sure to go around the table in a circular manner, don't ever call on the same agent twice in a row."

    available_agents = ["kweku", "invention-peddler", "banny"]
    session_id = "6906824cc9ab119e414c1f97"
    scenario = "Invention Peddler, Banny and Kweku are playing a game. The first one of them generates an image of anything they feel like. Then they take turns taking the last image made and modifying it somehow to make it more absurd. Each one more absurd than the last. Always use the last image made as the input image."

    available_agents = ["banny", "mechanical_duck", "abraham", "kweku"]
    session_id = "690694f6223d5f30d7986867"
    scenario = """Here's the game.

Banny is the Dungeon Master.

Abraham, Mechanical Duck, and Kweku are the players.

Each round of the game goes like this:
- Banny comes up with a scenario
- The players each take turns responding to the scenario, maybe with an image.
- Banny then tells them the outcome of their actions, and eliminates one of the players, and then comes up with a new scenario for the remaining players.
- This continues until there is one player left, and that player is declared winner, and asked to come up with a triumphant image of themselves winning the D&D challenge.
"""

    user = get_my_eden_user()
    
    agents = ""
    for agent_ in available_agents:
        agent = Agent.load(agent_)
        agents += f"  <Agent name=\"{agent.username}\" description=\"{agent.description}\" />\n"
    
    conductor_message = conductor_template.render(
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        agents=agents
    )
    
    messages = ChatMessage.find({"session": ObjectId(session_id)})

    class ConductorResponse(BaseModel):
        """Form an intention for the next speaker"""    

        speaker: Literal[*available_agents] = Field(description="The speaker who should speak next")
        hint: Optional[str] = Field(description="A hint to the speaker to keep turn. Turn constraints/budgets/phase reminders **only**.")

    # Build LLM context with custom tools
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message), 
            *messages,
            ChatMessage(role="user", content="<Task>Determine who should speak next, and issue a conservative hint if necessary.</Task>")
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            response_format=ConductorResponse
        ),
    )


    # print("=================")
    # for message in messages:
    #     print(message.role, message.content)
    #     print("---")
    # print("=================")

    # raise Exception("Stop here")


    # Do a single turn prompt with forced tool usage
    response = await async_prompt(context)
    # print(response)
    output = ConductorResponse(**json.loads(response.content))

    agent = Agent.load(output.speaker)
    hint = output.hint
    user_id = str(user.id)
    agent_id = str(agent.id)


    # print("selected agent: ", output.speaker)
    # print("hint: ", output.hint)

    await remote_prompt_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=user_id,
        content=f"",
        attachments=[],
        extra_tools=[],
    )










async def run_automatic_session(session_id: str):    
    session = Session.from_mongo(session_id)
    while True:
        session.reload()
        if session.status != "active":
            break
        await run_automatic_session_step(session)


async def run_automatic_session_step(session: Session):    
    agents = [Agent.from_mongo(a) for a in session.agents]
    agents = {agent.username: agent for agent in agents}    
    agent_str = ""
    for agent in agents.values():
        agent_str += f"  <Agent name=\"{agent.username}\" description=\"{agent.description}\" />\n"

    conductor_message = conductor_template.render(
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        agents=agent_str
    )

    messages = ChatMessage.find({"session": session.id})

    class ConductorResponse(BaseModel):
        """Form an intention for the next speaker"""    

        speaker: Literal[*agents.keys()] = Field(description="The speaker who should speak next")
        hint: Optional[str] = Field(description="A hint to the speaker to keep turn. Turn constraints/budgets/phase reminders **only**.")

    # Build LLM context with custom tools
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message), 
            *messages,
            ChatMessage(role="user", content="<Task>Determine who should speak next, and issue a conservative hint if necessary.</Task>")
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            response_format=ConductorResponse
        ),
    )

    response = await async_prompt(context)
    output = ConductorResponse(**json.loads(response.content))

    actor = Agent.load(output.speaker)

    await remote_prompt_session(
        session_id=str(session.id),
        agent_id=str(actor.id),
        user_id=str(session.owner),
        content=f"",
        attachments=[],
        extra_tools=[],
    )


























if __name__ == "__main__":
    # while True:
    #     asyncio.run(conductor())
    # asyncio.run(test())
    asyncio.run(run_automatic_session("690a76002df74800fac63c5e"))








