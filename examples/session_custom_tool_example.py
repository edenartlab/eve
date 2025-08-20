import asyncio
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput, LLMConfig
from eve.agent.session.session import add_user_message, build_llm_context, async_prompt_session
from eve.auth import get_my_eden_user
from eve.agent import Agent
from eve.tool import Tool
from eve.tools.tool_handlers import handlers


# Define a custom tool as a pydantic model
class EdenDescription(BaseModel):
    """A tool to structure a description of Eden"""    
    description: str = Field(description="A description of Eden")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment")

async def custom_handler(
    args: Dict[str, Any], user: str = None, agent: str = None
) -> Dict[str, Any]:
    result = f"Eden is {args['description']} and its sentiment {args['sentiment']}"
    return {"output": result}


# Create and register the tool
custom_tool = Tool.from_pydantic(EdenDescription)
handlers[custom_tool.key] = custom_handler


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
        content="Describe Eden and its sentiment. Use the EdenDescription tool."
    )

    # Create context and add custom tool
    context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=LLMConfig(model="gpt-4o-mini"),

        # insert custom tool here
        custom_tools={custom_tool.key: custom_tool},
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
        

if __name__ == "__main__":
    asyncio.run(example_session())