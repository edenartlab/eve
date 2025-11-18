import asyncio
from fastapi import BackgroundTasks
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput, LLMConfig
from eve.agent.session.session import add_chat_message, build_llm_context, async_prompt_session
from eve.auth import get_my_eden_user
from eve.agent import Agent


async def example_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("eve")

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
        role="user",
        content="What is Eden?"
    )

    # Create context
    prompt_context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=LLMConfig(model="gpt-4o-mini")
    )

    await add_chat_message(session, prompt_context)

    # Run session
    llm_context = await build_llm_context(
        session, 
        agent, 
        prompt_context, 
    )
    
    # Execute the prompt session
    async for _ in async_prompt_session(
        session, llm_context, agent, context=prompt_context
    ):
        pass
    
    # it should now be available under your sessions with Eve
    

if __name__ == "__main__":
    asyncio.run(example_session())
