"""Conductor system for automatic session orchestration.

The Conductor uses an LLM to determine which agent should speak next
in an automatic multi-agent session.
"""

import json
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger
from pydantic import BaseModel, Field

from eve.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.conductor_template import conductor_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext, Session


async def conductor_select_actor(session: Session) -> Agent:
    """Use LLM to determine which agent should speak next in an automatic session.

    Args:
        session: The session to select an actor for

    Returns:
        The Agent that should speak next
    """
    logger.info(f"[CONDUCTOR] conductor_select_actor called for session {session.id}")
    logger.info(f"[CONDUCTOR] Session agents: {session.agents}")

    # Load all agents in the session
    agents = {
        agent.username: agent for agent in [Agent.from_mongo(a) for a in session.agents]
    }
    logger.info(f"[CONDUCTOR] Loaded agents: {list(agents.keys())}")

    if not agents:
        logger.error("[CONDUCTOR] Session has no agents!")
        raise ValueError("Session has no agents")

    if len(agents) == 1:
        # Only one agent, no need for conductor
        agent = list(agents.values())[0]
        logger.info(f"[CONDUCTOR] Only one agent, returning {agent.username}")
        return agent

    # Build agent description string for conductor prompt
    agent_str = ""
    for agent in agents.values():
        agent_str += (
            f'  <Agent name="{agent.username}" description="{agent.description}" />\n'
        )
    logger.info(f"[CONDUCTOR] Agent descriptions:\n{agent_str}")

    # Render conductor system message
    conductor_message = conductor_template.render(
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        agents=agent_str,
    )
    logger.info(
        f"[CONDUCTOR] Conductor system message rendered ({len(conductor_message)} chars)"
    )

    # Get conversation history
    messages = list(ChatMessage.find({"session": session.id}))
    logger.info(f"[CONDUCTOR] Found {len(messages)} messages in conversation history")

    # Create dynamic response model with valid agent names
    # Using a simpler approach since Literal with dynamic values is tricky
    class ConductorResponse(BaseModel):
        """Conductor's decision on next speaker"""

        speaker: str = Field(
            description="The username of the agent who should speak next"
        )
        hint: Optional[str] = Field(
            default=None,
            description="A hint to the speaker. Turn constraints/budgets/phase reminders only.",
        )

    # Build LLM context
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message),
            *messages,
            ChatMessage(
                role="user",
                content="<Task>Determine who should speak next, and issue a conservative hint if necessary.</Task>",
            ),
        ],
        config=LLMConfig(model="claude-sonnet-4-5", response_format=ConductorResponse),
    )
    logger.info(f"[CONDUCTOR] Built LLM context with {len(context.messages)} messages")
    logger.info("[CONDUCTOR] Calling async_prompt...")

    # Get conductor's decision
    response = await async_prompt(context)
    logger.info(f"[CONDUCTOR] Got response: {response.content}")
    output = ConductorResponse(**json.loads(response.content))
    logger.info(
        f"[CONDUCTOR] Parsed response: speaker={output.speaker}, hint={output.hint}"
    )

    # Validate speaker is a valid agent
    if output.speaker not in agents:
        logger.error(
            f"[CONDUCTOR] Invalid speaker '{output.speaker}', valid agents: {list(agents.keys())}"
        )
        raise ValueError(
            f"Conductor selected invalid speaker '{output.speaker}'. "
            f"Valid agents: {list(agents.keys())}"
        )

    logger.info(f"[CONDUCTOR] Selected actor: {output.speaker}")
    return agents[output.speaker]
