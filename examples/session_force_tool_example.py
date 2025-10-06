import asyncio
import json
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput, LLMConfig
from eve.agent.session.session import add_chat_message, build_llm_context, async_prompt_session
from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.tool import Tool


# Create and register the tool
tool = Tool.load("discord_post")
tool.parameters["channel_id"].update({"default": "1181679778651181067", "hide_from_agent": True})



async def example_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("verdelis")

    # Create session request
    request = PromptSessionRequest(
        user_id=str(user.id),
        creation_args=SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(agent.id)],
            title="Example session"
        )
    )

    # Setup session
    session = setup_session(background_tasks, request.session_id, request.user_id, request)

    # Create message
    message = ChatMessageRequestInput(
        content="Post 'yes, it was that day, I say, oy vey' to discord channel '1181679778651181067'."
    )

    # Create context and add custom tool
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=LLMConfig(model="gpt-5"),

        # insert custom tool here
        tools={tool.key: tool},
        tool_choice=tool.key,
    )

    await add_chat_message(session, context)

    # Run session
    context = await build_llm_context(
        session, 
        agent, 
        context, 
    )

    # Execute the prompt session
    async for _ in async_prompt_session(session, context, agent):
        pass
        

if __name__ == "__main__":
    asyncio.run(example_session())