"""Conductor system for session orchestration.

The Conductor uses an LLM to determine which agent should speak next.
Two modes are available:
- Automatic mode: Must select an agent (for automatic sessions)
- Natural mode: Can decline to select (for natural sessions without mentions)
"""

import json
from datetime import datetime
from typing import List, Optional

import pytz
from loguru import logger
from pydantic import BaseModel, Field

from eve.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.conductor_template import conductor_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext, Session

# Number of recent messages to consider for conductor decisions
CONDUCTOR_MESSAGE_LOOKBACK = 10


def _build_agent_descriptions(agents: dict) -> str:
    """Build XML-formatted agent descriptions for conductor prompt."""
    agent_str = ""
    for agent in agents.values():
        agent_str += (
            f'  <Agent name="{agent.username}" description="{agent.description}" />\n'
        )
    return agent_str


def _get_recent_messages(
    session: Session, limit: int = CONDUCTOR_MESSAGE_LOOKBACK
) -> List[ChatMessage]:
    """Get the most recent messages from the session."""
    messages = list(ChatMessage.find({"session": session.id}))
    # Sort by creation time and take the last N
    messages = sorted(
        messages, key=lambda m: m.createdAt if m.createdAt else datetime.min
    )
    return messages[-limit:] if len(messages) > limit else messages


def _format_messages_for_conductor(
    messages: List[ChatMessage], agents: dict
) -> List[ChatMessage]:
    """Format messages with sender names visible for conductor decisions.

    The conductor needs to see WHO sent each message to make good turn-taking
    decisions. This adds sender attribution to message content.
    """
    formatted = []
    for msg in messages:
        # Find sender name
        sender_name = None
        if msg.sender:
            for username, agent in agents.items():
                if agent.id == msg.sender:
                    sender_name = username
                    break

        # Create a copy with sender attribution in content
        # Skip name prefix for messages wrapped in SystemMessage tags
        is_system_message = (
            msg.content
            and msg.content.startswith("<SystemMessage>")
            and msg.content.endswith("</SystemMessage>")
        )

        logger.info(
            f"[CONDUCTOR] is_system_message: {is_system_message}: {msg.content}"
        )

        if sender_name and msg.content and not is_system_message:
            attributed_content = f"[{sender_name}]: {msg.content}"
        else:
            attributed_content = msg.content or ""

        # Use user role so LLM sees it as conversation history
        formatted.append(
            ChatMessage(
                role="user",
                content=attributed_content,
                name=sender_name if not is_system_message else None,
            )
        )
    return formatted


async def conductor_select_actor(session: Session) -> Agent:
    """Select next speaker for automatic sessions. Must select an agent.

    Used in automatic sessions where the conversation must continue.
    The conductor analyzes the conversation and picks who should speak next.

    Args:
        session: The session to select an actor for

    Returns:
        The Agent that should speak next

    Raises:
        ValueError: If no agents in session or invalid agent selected
    """
    logger.info(f"[CONDUCTOR] Automatic mode for session {session.id}")

    # Load all agents
    agents = {
        agent.username: agent for agent in [Agent.from_mongo(a) for a in session.agents]
    }
    logger.info(f"[CONDUCTOR] Available agents: {list(agents.keys())}")
    logger.info(
        f"[CONDUCTOR] Session context: {session.context[:100] + '...' if session.context and len(session.context) > 100 else session.context or 'NONE'}"
    )
    logger.info(f"[CONDUCTOR] Last actor: {session.last_actor_id}")

    if not agents:
        raise ValueError("Session has no agents")

    if len(agents) == 1:
        agent = list(agents.values())[0]
        logger.info(f"[CONDUCTOR] Single agent: {agent.username}")
        return agent

    # Build conductor prompt
    agent_str = _build_agent_descriptions(agents)
    conductor_message = conductor_template.render(
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        agents=agent_str,
        context=session.context or None,
    )

    # Get recent messages and format with sender names
    raw_messages = _get_recent_messages(session)
    messages = _format_messages_for_conductor(raw_messages, agents)
    logger.info(f"[CONDUCTOR] Using {len(messages)} recent messages")

    # Log last few messages for debugging
    for msg in messages[-3:]:
        logger.info(
            f"[CONDUCTOR] Recent: {msg.content[:80] if msg.content else '(empty)'}..."
        )

    # Analyze recent speaker pattern for the task prompt
    recent_speakers = []
    for msg in raw_messages[-5:]:
        if msg.sender:
            for username, agent in agents.items():
                if agent.id == msg.sender:
                    recent_speakers.append(username)
                    break

    speaker_summary = ", ".join(recent_speakers) if recent_speakers else "none yet"

    class ConductorResponse(BaseModel):
        """Conductor's decision on next speaker"""

        reasoning: str = Field(description="Brief analysis of who should speak and why")
        speaker: str = Field(
            description="The username of the agent who should speak next"
        )
        hint: Optional[str] = Field(
            default=None,
            description="Optional hint for the speaker (constraints/reminders only)",
        )

    task_prompt = f"""<Task>
Analyze this conversation and determine who should speak next.

Recent speaker order (last 5): {speaker_summary}

Guidelines:
1. If there's a defined structure (e.g., game master → players → game master), follow it
2. Avoid having the same agent speak twice in a row unless the context requires it
3. If agents have different roles (e.g., host vs contestants), respect the turn-taking rules
4. Ensure all agents get fair participation over time
5. Consider who the conversation is naturally addressed to

First explain your reasoning, then select the speaker.
</Task>"""

    llm_context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message),
            *messages,
            ChatMessage(role="user", content=task_prompt),
        ],
        config=LLMConfig(model="claude-sonnet-4-5", response_format=ConductorResponse),
    )

    response = await async_prompt(llm_context)
    output = ConductorResponse(**json.loads(response.content))
    logger.info(f"[CONDUCTOR] Reasoning: {output.reasoning}")
    logger.info(f"[CONDUCTOR] Selected: {output.speaker}, hint: {output.hint}")

    if output.speaker not in agents:
        raise ValueError(
            f"Invalid speaker '{output.speaker}'. Valid: {list(agents.keys())}"
        )

    return agents[output.speaker]


async def conductor_select_actor_natural(session: Session) -> Optional[Agent]:
    """Select next speaker for natural sessions. May return None.

    Used in natural sessions when no agent is explicitly mentioned.
    The conductor uses a conservative approach:
    - Select an agent if they are indirectly referenced
    - Select an agent if the message strongly implies their interest
    - Return None if neither condition is met (conversation terminates)

    Args:
        session: The session to evaluate

    Returns:
        The Agent that should respond, or None if no response needed
    """
    logger.info(f"[CONDUCTOR] Natural mode for session {session.id}")

    # Load all agents
    agents = {
        agent.username: agent for agent in [Agent.from_mongo(a) for a in session.agents]
    }

    if not agents:
        return None

    if len(agents) == 1:
        # Single agent natural session - let them decide implicitly
        # (For single agent, we default to responding like passive 1:1)
        return list(agents.values())[0]

    # Build conductor prompt with conservative instructions
    agent_str = _build_agent_descriptions(agents)
    conductor_message = conductor_template.render(
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        agents=agent_str,
        context=session.context if hasattr(session, "context") else None,
    )

    # Get recent messages
    messages = _get_recent_messages(session)
    logger.info(f"[CONDUCTOR] Using {len(messages)} recent messages")

    class NaturalConductorResponse(BaseModel):
        """Conductor's decision for natural sessions"""

        should_respond: bool = Field(
            description="True if an agent should respond, False if no response is needed"
        )
        speaker: Optional[str] = Field(
            default=None,
            description="Username of agent to respond (only if should_respond is True)",
        )
        reasoning: str = Field(
            description="Brief explanation of why this agent should (or shouldn't) respond"
        )

    conservative_prompt = """<Task>
Analyze the most recent message and decide if any agent should respond.

Be CONSERVATIVE - only select an agent if:
1. The message indirectly references or addresses them (by topic, expertise, or context)
2. The message strongly implies the agent's interest or involvement
3. The conversation flow naturally calls for their input

If the message is general, ambiguous, or doesn't clearly warrant a specific agent's response, set should_respond to False.

Do NOT select an agent just because they could potentially contribute - there must be a clear reason.
</Task>"""

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message),
            *messages,
            ChatMessage(role="user", content=conservative_prompt),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5", response_format=NaturalConductorResponse
        ),
    )

    response = await async_prompt(context)
    output = NaturalConductorResponse(**json.loads(response.content))
    logger.info(
        f"[CONDUCTOR] Natural decision: respond={output.should_respond}, speaker={output.speaker}, reason={output.reasoning}"
    )

    if not output.should_respond:
        return None

    if output.speaker and output.speaker in agents:
        return agents[output.speaker]

    # Fallback: if should_respond but no valid speaker, return None
    logger.warning(
        f"[CONDUCTOR] should_respond=True but invalid speaker: {output.speaker}"
    )
    return None
