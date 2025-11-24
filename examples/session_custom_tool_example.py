import asyncio
from typing import Any, Dict, Literal

from fastapi import BackgroundTasks
from pydantic import BaseModel, Field

from eve.agent import Agent
from eve.agent.session.context import add_chat_message, build_llm_context
from eve.agent.session.models import ChatMessageRequestInput, LLMConfig
from eve.agent.session.runtime import async_prompt_session
from eve.agent.session.service import create_prompt_session_handle
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.auth import get_my_eden_user
from eve.tool import Tool, ToolContext


# Define a custom tool as a pydantic model
class EdenDescription(BaseModel):
    """A tool to structure a description of Eden"""

    description: str = Field(description="A description of Eden")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment"
    )


async def custom_handler(tool_context: ToolContext) -> Dict[str, Any]:
    result = f"Eden is {tool_context.args['description']} and its sentiment {tool_context.args['sentiment']}"
    return {"output": result}


# Create and register the tool
custom_tool = Tool.register_new(EdenDescription, custom_handler)


async def example_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("eve")

    message = ChatMessageRequestInput(
        content="Describe Eden and its sentiment. Use the EdenDescription tool."
    )
    llm_config = LLMConfig(model="gpt-4o-mini")

    # Create session request
    request = PromptSessionRequest(
        user_id=str(user.id),
        message=message,
        llm_config=llm_config,
        creation_args=SessionCreationArgs(
            owner_id=str(user.id), agents=[str(agent.id)], title="Example session"
        ),
    )

    handle = create_prompt_session_handle(request, background_tasks)
    session = handle.session
    prompt_context = handle.context
    prompt_context.extra_tools = {custom_tool.key: custom_tool}

    await add_chat_message(session, prompt_context)

    # Run session
    llm_context = await build_llm_context(
        session,
        agent,
        prompt_context,
    )

    # Execute the prompt session
    async for _ in async_prompt_session(
        session, llm_context=llm_context, agent=agent, context=prompt_context
    ):
        pass


if __name__ == "__main__":
    asyncio.run(example_session())
