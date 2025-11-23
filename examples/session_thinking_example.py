import asyncio
from fastapi import BackgroundTasks
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.api.handlers import setup_session
from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput, LLMConfig, LLMThinkingSettings
from eve.agent.session.session import add_chat_message, build_llm_context, async_prompt_session
from eve.auth import get_my_eden_user
from eve.agent import Agent



async def example_thinking_session():
    background_tasks = BackgroundTasks()

    user = get_my_eden_user()
    agent = Agent.load("eve")

    # Create session request
    request = PromptSessionRequest(
        user_id=str(user.id),
        creation_args=SessionCreationArgs(
            owner_id=str(user.id),
            agents=[str(agent.id)],
            title="Thinking Model Example Session"
        )
    )

    # Setup session
    session = setup_session(background_tasks, request.session_id, request.user_id, request)

    # Create message with a complex reasoning problem
    message = ChatMessageRequestInput(
        content="""You are tasked with solving a complex logical puzzle that requires deep reasoning.

        Problem: The Tower of Hanoi with a twist
        
        You have 4 towers (A, B, C, D) and 5 disks of different sizes (1=smallest, 5=largest).
        All disks start on tower A in order (1 on top, 5 on bottom).
        
        Rules:
        1. Only one disk can be moved at a time
        2. A disk can only be placed on top of a larger disk or on an empty tower
        3. You can only move the top disk from any tower
        4. Tower D has a special constraint: once a disk is placed on D, it cannot be moved until all smaller disks are removed from D
        
        Goal: Move all disks to tower C in the same order (1 on top, 5 on bottom)
        
        Please provide:
        1. A step-by-step solution with detailed reasoning for each move
        2. Mathematical analysis of why this solution is optimal
        3. How the special constraint on tower D affects the strategy compared to classic Tower of Hanoi
        4. The minimum number of moves required and proof of optimality
        
        Think through this carefully and show your complete reasoning process."""
    )

    message = ChatMessageRequestInput(
        content="""What's the best college to go to for an introvert? Think step by step."""
    )

    # Create context with thinking model configuration
    llm_config = LLMConfig(
        model="claude-sonnet-4-5-20250929",
        llm_settings=LLMThinkingSettings(
            policy="auto",
            effort_instructions="Use low when I ask you to think about Hanoi Problem, but high for anything else, especially when I ask you to think about the best college to go to for an introvert."
        )
        # todo: fix this
        # thinking=True,
        # thinking_budget_tokens=10000
    )
    
    prompt_context = PromptSessionContext(
        session=session,
        initiating_user_id=request.user_id,
        message=message,
        llm_config=llm_config
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
    asyncio.run(example_thinking_session())
