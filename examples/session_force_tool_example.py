import asyncio

from fastapi import BackgroundTasks

from eve.agent import Agent
from eve.agent.session.context import add_chat_message, build_llm_context
from eve.agent.session.models import ChatMessageRequestInput, LLMConfig
from eve.agent.session.runtime import async_prompt_session
from eve.agent.session.service import create_prompt_session_handle
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.auth import get_my_eden_user
from eve.tool import Tool

# Create and register the tool
tool = Tool.load("discord_post")
tool.parameters["channel_id"].update(
    {"default": "1181679778651181067", "hide_from_agent": True}
)


async def example_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("verdelis")

    message = ChatMessageRequestInput(
        content="Post 'yes, it was that day, I say, oy vey' to discord channel '1181679778651181067'."
    )
    llm_config = LLMConfig(model="gpt-5")

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
    context = handle.context
    context.tools = {tool.key: tool}
    context.tool_choice = tool.key

    await add_chat_message(session, context)

    # Run session
    context = await build_llm_context(
        session,
        agent,
        context,
    )

    # Execute the prompt session
    async for _ in async_prompt_session(
        session, llm_context=context, agent=agent, context=handle.context
    ):
        pass


if __name__ == "__main__":
    asyncio.run(example_session())
