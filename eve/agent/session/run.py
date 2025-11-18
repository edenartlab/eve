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
    content: Optional[str] = None,
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
        content=content or "",
        attachments=attachments,
    )

    # Build context
    prompt_context = PromptSessionContext(
        session=session,
        initiating_user_id=str(user.id),
        message=new_message,
        llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
    )

    if extra_tools:
        prompt_context.extra_tools = {k: Tool.load(k) for k in extra_tools}

    # Add message to session
    if content or attachments:
        await add_chat_message(session, prompt_context)

    # Build LLM context and prompt
    llm_context = await build_llm_context(
        session,
        agent,
        prompt_context,
        trace_id=str(uuid.uuid4()),
    )

    # Run the prompt
    async for m in async_prompt_session(
        session, llm_context, agent, context=prompt_context
    ):
        pass

    logger.info(f"Remote prompt completed for session {session_id}")





async def conductor():
    
    # available_agents = ["iannis", "glitch_gigabrain", "dadagan", "xander2"]
    available_agents = ["iannis", "glitch_gigabrain", "plantoid-49"]
    session_id = "690b68d99aa43032fcc0349e"
    
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
        content=None,
        attachments=[],
        extra_tools=[],
    )


async def run_automatic_session(session_id: str):    
    session = Session.from_mongo(session_id)
    
    if session.session_type != "automatic":
        raise ValueError("Session is not an automatic session")

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

        speaker: Literal[*agents.keys()] = Field(
            description="The speaker who should speak next"
        )
        hint: Optional[str] = Field(
            description="A hint to the speaker to keep turn. Turn constraints/budgets/phase reminders **only**."
        )

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
        content=None,
        attachments=[],
        extra_tools=[],
    )


























# if __name__ == "__main__":
#     while True:
#         asyncio.run(conductor())
    # asyncio.run(test())
    # asyncio.run(run_automatic_session("690a76002df74800fac63c5e"))







