import asyncio
from datetime import datetime

import pytz

from eve.agent.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.system_template import SYSTEM_TEMPLATE as system_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext


async def example_prompt():
    agent = Agent.load("eve")

    system_message = system_template.render(
        name=agent.name,
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        persona=agent.persona,
        tools=None,
    )

    # Build LLM context with custom tools
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user", content="What is your name? Tell me about yourself."
            ),
        ],
        config=LLMConfig(model="gpt-4o-mini"),
    )

    # Do a single turn prompt with forced tool usage
    await async_prompt(context)

    # print(response)


if __name__ == "__main__":
    asyncio.run(example_prompt())
