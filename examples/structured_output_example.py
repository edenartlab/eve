import asyncio
import json
from datetime import datetime
from typing import Literal

import pytz
from pydantic import BaseModel, Field

from eve.agent.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.system_template import SYSTEM_TEMPLATE as system_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext


# Define a custom tool as a pydantic model
class EdenDescription(BaseModel):
    """A tool to structure a description of Eden"""

    description: str = Field(description="A description of Eden")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment"
    )
    reason_to_use: str = Field(description="The reason to use Eden")


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
            ChatMessage(role="user", content="How would you describe Eden?"),
        ],
        config=LLMConfig(model="gpt-4o-mini", response_format=EdenDescription),
    )

    # Do a single turn prompt with forced tool usage
    response = await async_prompt(context)

    print(response)

    output = EdenDescription(**json.loads(response.content))

    print(output)


if __name__ == "__main__":
    asyncio.run(example_prompt())
