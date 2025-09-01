import asyncio

from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.llm.config import LLMConfig
from eve.session.session import add_user_message
from eve.session.run import async_prompt_session
from eve.context.context import build_llm_context

from eve.session2 import setup_session

from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput


async def example_session():
    # raise NotImplementedError("async_prompt_session is not implemented ^^$#%^#$^#$")
    # background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("eve")

    # Create session request
    request = PromptSessionRequest(
        user_id=str(user.id),
        creation_args=SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(agent.id)],
            title="Example session 5333"
        )
    )

    # Setup session
    # session = setup_session(background_tasks, request.session_id, request.user_id, request)
    session = setup_session(request.session_id, request.user_id, request)

    # Create message
    message = ChatMessageRequestInput(
        role="user",
        content="What is your name? What do you do? When was Eden created? Make a picture of a cat. Use the 'create' tool. Do whatever you want, I trust you."
    )

    # Create context
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=LLMConfig(model="gpt-4o-mini")
    )

    add_user_message(session, context)

    # Run session
    context = await build_llm_context(
        session, 
        agent, 
        context, 
    )
    
    # Execute the prompt session
    async for _ in async_prompt_session(session, context, agent):
        pass
    
    # it should now be available under your sessions with Eve
    

if __name__ == "__main__":
    asyncio.run(example_session())