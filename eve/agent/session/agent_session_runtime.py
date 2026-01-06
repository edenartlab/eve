"""Agent session runtime for multi-agent orchestration.

Handles the private workspace flow:
1. Build LLM context (messages are already distributed via real-time distribution)
2. Run LLM + tool processing until agent uses 'chat' tool

This module bridges the gap between the parent chatroom session and each
agent's private workspace (agent_session). Messages from other agents are
distributed in real-time to agent_sessions, so no bulk sync is needed.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.context import build_agent_session_llm_context
from eve.agent.session.instrumentation import PromptSessionInstrumentation
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    PromptSessionContext,
    Session,
    UpdateType,
)
from eve.agent.session.runtime import PromptSessionRuntime


class VoteTask:
    """A minimal Task-like object for vote collection."""

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        choice = args.get("choice")
        reasoning = args.get("reasoning", "")  # Default to empty string if not provided

        # Build result dict for UI display
        ui_result = {"vote": choice}
        if reasoning:
            ui_result["reasoning"] = reasoning

        self.result = {
            "status": "completed",
            # Nested "result" key for tool_call.result (displayed in UI)
            # Must be a List[Dict] to match ToolCall.result type
            "result": [ui_result],
            "output": f"Vote recorded: {choice}",
            # Top-level for easy extraction in run_agent_vote_turn
            "choice": choice,
            "reasoning": reasoning,
        }
        self.status = "completed"
        self.handler_id = "vote_task"
        self.mock = False

    def reload(self):
        """No-op reload since we don't persist to DB."""
        pass


class VoteTool:
    """A minimal tool class for vote collection.

    This implements just enough of the Tool interface to work with
    the LLM context formatting and runtime. The "handler" just returns
    the vote - no side effects needed.
    """

    def __init__(self, choices: List[str], reasoning_required: bool = False):
        self.name = "vote"
        self.key = "vote"
        self.choices = choices
        self.reasoning_required = reasoning_required
        self.visible = True
        self._handler_cache: Dict[str, Any] = {}

        if reasoning_required:
            self.description = "Cast your vote with reasoning. You MUST use this tool to submit your vote."
        else:
            self.description = (
                "Cast your vote. You MUST use this tool to submit your vote."
            )

    def openai_schema(self, exclude_hidden: bool = False) -> dict:
        """Return the OpenAI tool schema for this vote tool."""
        properties = {
            "choice": {
                "type": "string",
                "enum": self.choices,
                "description": f"Your vote. Must be exactly one of: {self.choices}",
            },
        }
        required = ["choice"]

        if self.reasoning_required:
            properties["reasoning"] = {
                "type": "string",
                "description": "Brief reasoning for your vote choice (1-2 sentences)",
            }
            required.append("reasoning")

        return {
            "type": "function",
            "function": {
                "name": "vote",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def async_start_task(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        message_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        mock: bool = False,
        public: bool = False,
        is_client_platform: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VoteTask:
        """Start the vote 'task' - returns immediately with the vote result."""
        return VoteTask(args or {})

    async def async_wait(self, task: VoteTask) -> Dict[str, Any]:
        """Wait for task completion - already complete, just return result."""
        return task.result


async def run_agent_session_turn(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
) -> None:
    """Run a single turn in an agent_session.

    This function:
    1. Loads the agent_session
    2. Builds LLM context (messages from other agents are already in agent_session
       via real-time distribution)
    3. Runs the prompt loop until the agent posts to chatroom

    The agent works privately until they call the chat tool,
    which posts to the parent session and distributes to other agent_sessions.

    Note: This uses PromptSessionRuntime directly (not orchestrator) because
    it requires specialized context building via build_agent_session_llm_context.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session
    """
    logger.info("[AGENT_SESSION] ========== run_agent_session_turn START ==========")
    logger.info(f"[AGENT_SESSION] Parent session: {parent_session.id}")
    logger.info(f"[AGENT_SESSION] Agent session ID: {agent_session_id}")
    logger.info(f"[AGENT_SESSION] Actor: {actor.username} (id={actor.id})")

    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        logger.error(f"[AGENT_SESSION] Agent session {agent_session_id} not found")
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(f"[AGENT_SESSION] Loaded agent_session: title='{agent_session.title}'")
    logger.info(
        f"[AGENT_SESSION] agent_session.context present: {bool(agent_session.context)}, "
        f"length: {len(agent_session.context) if agent_session.context else 0}"
    )
    if agent_session.context:
        logger.info(
            f"[AGENT_SESSION] Context preview: {agent_session.context[:150]}..."
        )

    # Generate session run ID for this turn
    session_run_id = str(uuid.uuid4())
    logger.info(f"[AGENT_SESSION] Session run ID: {session_run_id}")

    # Create instrumentation for Langfuse/Sentry tracing
    instrumentation = PromptSessionInstrumentation(
        session_id=str(agent_session.id),
        session_run_id=session_run_id,
        agent_id=str(actor.id),
        user_id=str(parent_session.owner),
        trace_name=f"agent_session_{actor.username}",
    )
    instrumentation.ensure_sentry_transaction(
        name=f"agent_session_{actor.username}",
        op="session.agent_turn",
    )
    logger.info(
        f"[AGENT_SESSION] Instrumentation created: trace_name={instrumentation.trace_name}"
    )

    # Build context for this turn
    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Build LLM context - messages from other agents are already in agent_session
    # via real-time distribution (no bulk sync needed)
    logger.info("[AGENT_SESSION] ===== Building LLM Context =====")
    logger.info("[AGENT_SESSION] Calling build_agent_session_llm_context...")
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
        instrumentation=instrumentation,
    )
    logger.info("[AGENT_SESSION] LLM context built successfully")

    # Run the prompt loop using the standard runtime
    # Note: Using PromptSessionRuntime directly for specialized agent session handling
    logger.info("[AGENT_SESSION] ===== Running PromptSessionRuntime =====")
    runtime = PromptSessionRuntime(
        session=agent_session,
        llm_context=llm_context,
        actor=actor,
        stream=False,
        is_client_platform=False,
        session_run_id=session_run_id,
        api_key_id=None,
        context=context,
        instrumentation=instrumentation,
    )

    posted_to_parent = False
    update_count = 0
    logger.info("[AGENT_SESSION] Starting runtime.run() loop...")

    async for update in runtime.run():
        update_count += 1
        logger.info(f"[AGENT_SESSION] Update #{update_count}: type={update.type}")

        # Check if agent posted to parent via chat tool
        if update.type == UpdateType.TOOL_COMPLETE and update.tool_name == "chat":
            posted_to_parent = True
            logger.info(f"[AGENT_SESSION] >>> {actor.username} posted to chatroom <<<")

    logger.info(f"[AGENT_SESSION] Runtime loop completed, {update_count} updates")

    if posted_to_parent:
        logger.info(
            f"[AGENT_SESSION] Turn completed: {actor.username} posted to chatroom"
        )
    else:
        logger.warning(
            f"[AGENT_SESSION] Turn completed: {actor.username} did NOT post to chatroom"
        )

    # Finalize instrumentation (flush to Langfuse)
    instrumentation.finalize(success=posted_to_parent)
    logger.info("[AGENT_SESSION] ========== run_agent_session_turn END ==========")


async def run_agent_session_turn_streaming(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
):
    """Streaming version of run_agent_session_turn.

    Yields SessionUpdates as they occur, allowing real-time streaming
    of the agent's work.

    Note: This uses PromptSessionRuntime directly (not orchestrator) because
    it requires specialized context building via build_agent_session_llm_context.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's private workspace session
        actor: The Agent who owns this agent_session

    Yields:
        SessionUpdate objects as the agent works
    """
    logger.info(
        "[AGENT_SESSION_STREAM] ========== run_agent_session_turn_streaming START =========="
    )
    logger.info(f"[AGENT_SESSION_STREAM] Parent session: {parent_session.id}")
    logger.info(f"[AGENT_SESSION_STREAM] Agent session ID: {agent_session_id}")
    logger.info(f"[AGENT_SESSION_STREAM] Actor: {actor.username} (id={actor.id})")

    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        logger.error(
            f"[AGENT_SESSION_STREAM] Agent session {agent_session_id} not found"
        )
        raise ValueError(f"Agent session {agent_session_id} not found")

    logger.info(
        f"[AGENT_SESSION_STREAM] Loaded agent_session: title='{agent_session.title}'"
    )

    session_run_id = str(uuid.uuid4())
    logger.info(f"[AGENT_SESSION_STREAM] Session run ID: {session_run_id}")

    # Create instrumentation for Langfuse/Sentry tracing
    instrumentation = PromptSessionInstrumentation(
        session_id=str(agent_session.id),
        session_run_id=session_run_id,
        agent_id=str(actor.id),
        user_id=str(parent_session.owner),
        trace_name=f"agent_session_{actor.username}",
    )
    instrumentation.ensure_sentry_transaction(
        name=f"agent_session_{actor.username}",
        op="session.agent_turn",
    )

    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
    )

    # Build LLM context - messages from other agents are already in agent_session
    # via real-time distribution (no bulk sync needed)
    logger.info("[AGENT_SESSION_STREAM] ===== Building LLM Context =====")
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
        instrumentation=instrumentation,
    )
    logger.info("[AGENT_SESSION_STREAM] LLM context built")

    # Run with streaming
    logger.info(
        "[AGENT_SESSION_STREAM] ===== Running PromptSessionRuntime (streaming) ====="
    )
    runtime = PromptSessionRuntime(
        session=agent_session,
        llm_context=llm_context,
        actor=actor,
        stream=True,  # Enable streaming
        is_client_platform=False,
        session_run_id=session_run_id,
        api_key_id=None,
        context=context,
        instrumentation=instrumentation,
    )

    posted_to_parent = False
    update_count = 0
    logger.info("[AGENT_SESSION_STREAM] Starting runtime.run() loop...")

    async for update in runtime.run():
        update_count += 1
        yield update

        if update.type == UpdateType.TOOL_COMPLETE and update.tool_name == "chat":
            posted_to_parent = True
            logger.info(
                f"[AGENT_SESSION_STREAM] >>> {actor.username} posted to chatroom <<<"
            )

    logger.info(
        f"[AGENT_SESSION_STREAM] Runtime loop completed, {update_count} updates yielded"
    )

    if posted_to_parent:
        logger.info(
            f"[AGENT_SESSION_STREAM] Turn completed: {actor.username} posted to chatroom"
        )
    else:
        logger.warning(
            f"[AGENT_SESSION_STREAM] Turn completed: {actor.username} did NOT post to chatroom"
        )

    # Finalize instrumentation (flush to Langfuse)
    instrumentation.finalize(success=posted_to_parent)
    logger.info(
        "[AGENT_SESSION_STREAM] ========== run_agent_session_turn_streaming END =========="
    )


async def run_agent_vote_turn(
    parent_session: Session,
    agent_session_id: ObjectId,
    actor: Agent,
    topic: str,
    choices: List[str],
    reasoning_required: bool = False,
) -> Tuple[str, str, str]:
    """Run a vote turn for an agent in their workspace session.

    This function:
    1. Posts a vote request to the agent's workspace
    2. Runs their turn with only the vote tool available (forced via tool_choice)
    3. Extracts and returns their vote

    The vote interaction is preserved in the agent's session history, so they
    can see "I was asked to vote on X and I voted for Y" in future turns.

    Args:
        parent_session: The parent chatroom session
        agent_session_id: The ObjectId of the agent's workspace session
        actor: The Agent who is voting
        topic: The question/topic to vote on
        choices: Available voting options
        reasoning_required: Whether voters must provide reasoning (default False)

    Returns:
        Tuple of (username, choice, reasoning)
    """
    logger.info("[AGENT_VOTE] ========== run_agent_vote_turn START ==========")
    logger.info(f"[AGENT_VOTE] Agent: {actor.username}, Topic: {topic[:50]}...")
    logger.info(
        f"[AGENT_VOTE] Choices: {choices}, reasoning_required={reasoning_required}"
    )

    agent_session = Session.from_mongo(agent_session_id)
    if not agent_session:
        logger.error(f"[AGENT_VOTE] Agent session {agent_session_id} not found")
        raise ValueError(f"Agent session {agent_session_id} not found")

    # Generate session run ID for this vote turn
    session_run_id = str(uuid.uuid4())

    # Create instrumentation
    instrumentation = PromptSessionInstrumentation(
        session_id=str(agent_session.id),
        session_run_id=session_run_id,
        agent_id=str(actor.id),
        user_id=str(parent_session.owner),
        trace_name=f"agent_vote_{actor.username}",
    )
    instrumentation.ensure_sentry_transaction(
        name=f"agent_vote_{actor.username}",
        op="session.agent_vote",
    )

    # Post vote request to agent's workspace
    if reasoning_required:
        vote_instruction = (
            "You MUST use the **vote** tool to cast your vote with reasoning."
        )
    else:
        vote_instruction = "You MUST use the **vote** tool to cast your vote."

    vote_request_content = f"""🗳️ VOTE REQUIRED

**Topic:** {topic}

**Available choices:**
{chr(10).join(f'• {c}' for c in choices)}

{vote_instruction}"""

    vote_request = ChatMessage(
        session=[agent_session_id],
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="user",
        content=vote_request_content,
    )
    vote_request.save()
    logger.info(f"[AGENT_VOTE] Posted vote request to {actor.username}'s workspace")

    # Create the vote tool
    vote_tool = VoteTool(choices, reasoning_required=reasoning_required)

    # Build context with only the vote tool, forced via tool_choice
    context = PromptSessionContext(
        session=agent_session,
        initiating_user_id=str(parent_session.owner),
        message=ChatMessageRequestInput(role="system", content=""),
        session_run_id=session_run_id,
        tools={"vote": vote_tool},  # Override tools - only vote tool
        tool_choice="vote",  # Force the vote tool
    )

    # Build LLM context - will use our tool override
    logger.info("[AGENT_VOTE] Building LLM context with vote tool override...")
    llm_context = await build_agent_session_llm_context(
        agent_session=agent_session,
        parent_session=parent_session,
        actor=actor,
        context=context,
        trace_id=session_run_id,
        instrumentation=instrumentation,
    )

    # Run the prompt loop
    logger.info("[AGENT_VOTE] Running PromptSessionRuntime for vote...")
    runtime = PromptSessionRuntime(
        session=agent_session,
        llm_context=llm_context,
        actor=actor,
        stream=False,
        is_client_platform=False,
        session_run_id=session_run_id,
        api_key_id=None,
        context=context,
        instrumentation=instrumentation,
    )

    # Collect the vote from the runtime
    vote_choice = None
    vote_reasoning = None

    async for update in runtime.run():
        logger.info(f"[AGENT_VOTE] Update: type={update.type}")

        # Look for the vote tool completion
        if update.type == UpdateType.TOOL_COMPLETE and update.tool_name == "vote":
            # Extract vote from tool result
            # SessionUpdate has 'result' not 'tool_result'
            if update.result:
                vote_choice = update.result.get("choice")
                vote_reasoning = update.result.get("reasoning", "")
                logger.info(
                    f"[AGENT_VOTE] {actor.username} voted '{vote_choice}': {vote_reasoning[:50]}..."
                )

    # Finalize instrumentation
    instrumentation.finalize(success=vote_choice is not None)

    if not vote_choice:
        logger.warning(f"[AGENT_VOTE] No vote received from {actor.username}")
        vote_choice = "ABSTAIN"
        vote_reasoning = "Agent did not cast a vote"

    logger.info("[AGENT_VOTE] ========== run_agent_vote_turn END ==========")

    return actor.username, vote_choice, vote_reasoning
