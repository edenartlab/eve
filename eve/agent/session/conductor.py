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


# =============================================================================
# Structured Output Schemas for Conductor Operations
# =============================================================================


class AgentContext(BaseModel):
    """Context generated for a single agent during initialization"""

    agent_username: str = Field(description="Username of the agent")
    context: str = Field(
        description="Personalized context including role, rules, secrets, goals"
    )


class ConductorInitResponse(BaseModel):
    """Conductor's initialization output - generates contexts for all agents"""

    shared_understanding: str = Field(
        description="Common knowledge all agents share (rules, setting, public info)"
    )
    agent_contexts: List[AgentContext] = Field(
        description="Per-agent personalized contexts"
    )
    finish_criteria: str = Field(
        description="Conditions under which the session should end"
    )


class ConductorTurnResponse(BaseModel):
    """Conductor's turn selection decision"""

    reasoning: str = Field(description="Brief analysis of who should speak and why")
    speaker: str = Field(description="Username of the agent who should speak next")
    hint: Optional[str] = Field(
        default=None,
        description="CONSERVATIVE: Only factual updates (budget, round, state). Never strategic advice.",
    )
    finish: bool = Field(
        default=False,
        description="True if session should end (goal reached or criteria met)",
    )


class ConductorFinishResponse(BaseModel):
    """Conductor's session summary when finishing"""

    summary: str = Field(
        description="Summary of session outcome, results, and key moments"
    )
    outcome: Optional[str] = Field(
        default=None, description="Winner, resolution, or final state"
    )


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


async def conductor_select_actor(
    session: Session,
) -> tuple[Agent, ConductorTurnResponse]:
    """Select next speaker for automatic sessions with finish control.

    Used in automatic sessions where the conversation must continue.
    The conductor analyzes the conversation, picks who should speak next,
    and can optionally end the session.

    Args:
        session: The session to select an actor for

    Returns:
        Tuple of (selected_agent, conductor_response with hint and finish flag)

    Raises:
        ValueError: If no agents in session or invalid agent selected
    """
    from eve.agent.session.models import EdenMessageType
    from eve.agent.session.setup import create_eden_message_json

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

    # For single agent sessions, return a simple response (no LLM call needed)
    if len(agents) == 1:
        agent = list(agents.values())[0]
        logger.info(f"[CONDUCTOR] Single agent: {agent.username}")
        response = ConductorTurnResponse(
            reasoning="Single agent session - no selection needed",
            speaker=agent.username,
            hint=None,
            finish=False,
        )
        return agent, response

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

    # Build budget info for the prompt
    budget_info = ""
    if session.budget:
        turns_spent = session.budget.turns_spent or 0
        turn_budget = session.budget.turn_budget
        if turn_budget:
            budget_info = f"Turn {turns_spent + 1} of {turn_budget}."

    task_prompt = f"""<Task>
Analyze this conversation and determine:
1. Who should speak next
2. Whether to provide a hint (RARE - only factual state updates)
3. Whether the session should finish

{budget_info}

Recent speaker order (last 5): {speaker_summary}

Guidelines:
1. If there's a defined structure (e.g., game master → players → game master), follow it
2. Avoid having the same agent speak twice in a row unless the context requires it
3. If agents have different roles (e.g., host vs contestants), respect the turn-taking rules
4. Ensure all agents get fair participation over time
5. Consider who the conversation is naturally addressed to

HINT GUIDELINES (use sparingly):
- GOOD: "Budget: $500 remaining", "Round 3 of 5", "Alice has folded"
- BAD: "Consider attacking Bob", "Don't trust Alice", "Try harder"
- Hints must be NEUTRAL, FACTUAL, and NOT strategic advice

FINISH: Set finish=true if the session's goals have been achieved, a clear conclusion
has been reached, or the conversation has naturally ended.

First explain your reasoning, then select the speaker.
</Task>"""

    llm_context = LLMContext(
        messages=[
            ChatMessage(role="system", content=conductor_message),
            *messages,
            ChatMessage(role="user", content=task_prompt),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5", response_format=ConductorTurnResponse
        ),
    )

    response = await async_prompt(llm_context)
    output = ConductorTurnResponse(**json.loads(response.content))
    logger.info(f"[CONDUCTOR] Reasoning: {output.reasoning}")
    logger.info(
        f"[CONDUCTOR] Selected: {output.speaker}, hint: {output.hint}, finish: {output.finish}"
    )

    if output.speaker not in agents:
        raise ValueError(
            f"Invalid speaker '{output.speaker}'. Valid: {list(agents.keys())}"
        )

    # Save as CONDUCTOR_TURN eden message
    create_eden_message_json(
        session_id=session.id,
        message_type=EdenMessageType.CONDUCTOR_TURN,
        content=output.model_dump_json(),
    )
    logger.info("[CONDUCTOR] Saved CONDUCTOR_TURN eden message")

    return agents[output.speaker], output


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


# =============================================================================
# Conductor Initialization - Generates per-agent contexts at session start
# =============================================================================

CONDUCTOR_INIT_TEMPLATE = """<AGENT_SPEC name="ConductorInit" version="1.0">
  <Summary>
    You are initializing a multi-agent session. Your job is to parse the user's scenario
    and generate personalized context for each participating agent.
  </Summary>

  <Responsibilities>
    1. Identify SHARED KNOWLEDGE: Rules, settings, and information all agents know
    2. Identify PER-AGENT SECRETS: Information only specific agents should know
    3. Define FINISH CRITERIA: When should this session conclude?
  </Responsibilities>

  <CriticalRules>
    - NEVER leak Agent A's secrets into Agent B's context
    - Keep each agent's context focused and relevant to THEIR role
    - Preserve the user's intent faithfully
    - Be conservative with secrets - only separate information that MUST be hidden
    - If no secrets are implied, all agents get similar contexts with role variations
  </CriticalRules>

  <Context>
    Current date: {current_date}
  </Context>

  <Agents>
{agents}
  </Agents>
</AGENT_SPEC>"""


async def conductor_initialize_session(
    session: Session,
    agents: dict,
) -> ConductorInitResponse:
    """Generate personalized contexts for each agent at session start.

    The conductor parses the user's scenario to identify:
    - Shared rules and knowledge
    - Per-agent roles, secrets, and objectives
    - Session finish criteria

    Args:
        session: The parent session being initialized
        agents: Dict mapping username to Agent object

    Returns:
        ConductorInitResponse with contexts for each agent
    """
    logger.info(f"[CONDUCTOR_INIT] Initializing session {session.id}")
    logger.info(f"[CONDUCTOR_INIT] Agents: {list(agents.keys())}")
    logger.info(
        f"[CONDUCTOR_INIT] User context: {session.context[:200] + '...' if session.context and len(session.context) > 200 else session.context or 'NONE'}"
    )

    # Build agent descriptions
    agent_str = _build_agent_descriptions(agents)

    # Build system message
    system_message = CONDUCTOR_INIT_TEMPLATE.format(
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        agents=agent_str,
    )

    # Build task prompt
    task_prompt = f"""<Task>
You are initializing a multi-agent session. Analyze the scenario and generate:

1. SHARED UNDERSTANDING: Common knowledge all agents share (rules, setting, public info)

2. PER-AGENT CONTEXTS: For each agent ({', '.join(agents.keys())}), generate a personalized context including:
   - Their role in the scenario
   - Any private information or secrets they alone know
   - Their objectives or win conditions
   - Relevant constraints or guidelines

3. FINISH CRITERIA: When should this session end? (e.g., "after 10 turns", "when a winner is declared", "when the task is complete")

CRITICAL RULES:
- Never leak Agent A's secrets into Agent B's context
- Keep contexts concise but complete
- Preserve the user's intent faithfully
- If no secrets are implied, all agents get similar contexts with role variations

User's Scenario:
{session.context or "No specific scenario provided. Generate a general collaborative discussion context."}
</Task>"""

    llm_context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=task_prompt),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5", response_format=ConductorInitResponse
        ),
    )

    response = await async_prompt(llm_context)
    output = ConductorInitResponse(**json.loads(response.content))

    logger.info(
        f"[CONDUCTOR_INIT] Generated {len(output.agent_contexts)} agent contexts"
    )
    logger.info(f"[CONDUCTOR_INIT] Finish criteria: {output.finish_criteria}")
    for ac in output.agent_contexts:
        logger.info(
            f"[CONDUCTOR_INIT] Context for {ac.agent_username}: {ac.context[:100]}..."
        )

    return output


# =============================================================================
# Conductor Finish - Generates session summary when ending
# =============================================================================

CONDUCTOR_FINISH_TEMPLATE = """<AGENT_SPEC name="ConductorFinish" version="1.0">
  <Summary>
    You are concluding a multi-agent session. Your job is to summarize what happened
    and declare any outcomes or results.
  </Summary>

  <Context>
    Current date: {current_date}
  </Context>

  <Agents>
{agents}
  </Agents>
</AGENT_SPEC>"""


async def conductor_finish_session(session: Session) -> ConductorFinishResponse:
    """Generate session summary and pause the session.

    Called when the conductor decides to end the session (finish=true)
    or when the turn budget is exhausted.

    Args:
        session: The session to finish

    Returns:
        ConductorFinishResponse with summary and outcome
    """
    logger.info(f"[CONDUCTOR_FINISH] Finishing session {session.id}")

    # Load agents
    agents = {
        agent.username: agent for agent in [Agent.from_mongo(a) for a in session.agents]
    }
    agent_str = _build_agent_descriptions(agents)

    # Get conversation history
    raw_messages = _get_recent_messages(session, limit=50)  # More messages for summary
    messages = _format_messages_for_conductor(raw_messages, agents)

    # Build system message
    system_message = CONDUCTOR_FINISH_TEMPLATE.format(
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        agents=agent_str,
    )

    task_prompt = """<Task>
The session is concluding. Generate a summary including:
1. What happened in this session (key events and discussions)
2. The outcome or resolution (if applicable - winners, decisions, conclusions)
3. Any notable moments or turning points

Be concise but comprehensive. This summary will be visible to all participants.
</Task>"""

    llm_context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            *messages,
            ChatMessage(role="user", content=task_prompt),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5", response_format=ConductorFinishResponse
        ),
    )

    response = await async_prompt(llm_context)
    output = ConductorFinishResponse(**json.loads(response.content))

    logger.info(f"[CONDUCTOR_FINISH] Summary: {output.summary[:200]}...")
    logger.info(f"[CONDUCTOR_FINISH] Outcome: {output.outcome}")

    return output
