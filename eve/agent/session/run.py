import json
import logging
import uuid
from datetime import datetime
from typing import List, Literal, Optional

import pytz
from pydantic import BaseModel, Field

from eve.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.conductor_template import conductor_template
from eve.agent.session.context import (
    add_chat_message,
    build_llm_context,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    PromptSessionContext,
    Session,
)
from eve.agent.session.runtime import async_prompt_session
from eve.tool import Tool
from eve.user import User

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
        agent_str += (
            f'  <Agent name="{agent.username}" description="{agent.description}" />\n'
        )

    conductor_message = conductor_template.render(
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        agents=agent_str,
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
            ChatMessage(
                role="user",
                content="<Task>Determine who should speak next, and issue a conservative hint if necessary.</Task>",
            ),
        ],
        config=LLMConfig(model="claude-sonnet-4-5", response_format=ConductorResponse),
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
