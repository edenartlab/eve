import asyncio
import pytz
from datetime import datetime
from eve.agent.agent import Agent
from eve.agent.session.models import ChatMessage, LLMContext, LLMConfig
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import system_template


async def example_prompt():
    agent = Agent.load("eve")
    
    system_message = system_template.render(
        name=agent.name,
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        description=agent.description,
        persona=agent.persona,
        tools=None
    )

    # Build LLM context with custom tools
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message), 
            ChatMessage(role="user", content="What is your name? Tell me about yourself.")
        ],
        config=LLMConfig(model="gpt-4o-mini"),
    )
    
    # Do a single turn prompt with forced tool usage
    response = await async_prompt(context)

    print(response)


if __name__ == "__main__":
    asyncio.run(example_prompt())
