import asyncio

from fastapi import BackgroundTasks

from eve.agent import Agent
from eve.agent.session.context import add_chat_message, build_llm_context
from eve.agent.session.models import ChatMessageRequestInput, LLMConfig
from eve.agent.session.runtime import async_prompt_session
from eve.agent.session.service import create_prompt_session_handle
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.auth import get_my_eden_user


async def example_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("eve")

    message = ChatMessageRequestInput(role="user", content="What is Eden?")
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

    # it should now be available under your sessions with Eve


if __name__ == "__main__":
    asyncio.run(example_session())
