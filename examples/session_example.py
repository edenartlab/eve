import asyncio
from fastapi import BackgroundTasks
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput
from eve.agent.session.session import run_prompt_session
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
        content="What is Eden?"
    )

    # Create context
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message
    )

    # Run session
    await run_prompt_session(context, background_tasks)

    # it should now be available under your sessions with Eve
    

if __name__ == "__main__":
    asyncio.run(example_session())