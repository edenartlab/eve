"""Handler for moderator conduct_vote tool.

This tool runs a vote among specified agents on a topic.
Voting happens in parallel batches of 4 for efficiency.

Each agent votes within their own workspace session, so the vote interaction
is preserved in their session history. They can see "I was asked to vote on X
and I voted for Y" in future turns.
"""

import asyncio
import json
from typing import Any, Dict, List

from bson import ObjectId
from loguru import logger

from eve.agent import Agent
from eve.agent.session.agent_session_runtime import run_agent_vote_turn
from eve.agent.session.models import (
    ChatMessage,
    EdenMessageData,
    EdenMessageType,
    Session,
)
from eve.tool import ToolContext

# Max number of agents to vote in parallel
VOTE_BATCH_SIZE = 4


async def handler(context: ToolContext) -> Dict[str, Any]:
    """Conduct a vote among agents.

    Each agent votes in their own workspace session:
    1. A vote request message is posted to their workspace
    2. Their turn runs with only the vote tool available (forced)
    3. Their vote is extracted and the interaction is preserved in their history

    Votes are collected in parallel batches of 4 for efficiency.

    Args:
        context: ToolContext containing:
            - args.topic: The question/topic to vote on
            - args.choices: Available voting options
            - args.voters: Usernames of agents who should vote
            - session: The moderator_session ID (which has a parent_session)

    Returns:
        Dict with vote tally and individual votes

    Raises:
        Exception: If validation fails or required data is missing
    """
    if not context.session:
        raise Exception("Session is required")

    # Get the moderator session
    moderator_session = Session.from_mongo(context.session)
    if not moderator_session:
        raise Exception(f"Moderator session {context.session} not found")

    if not moderator_session.parent_session:
        raise Exception(
            "This tool can only be used from a moderator_session with a parent. "
            "The current session has no parent_session."
        )

    # Get the parent session
    parent_session = Session.from_mongo(moderator_session.parent_session)
    if not parent_session:
        raise Exception(f"Parent session {moderator_session.parent_session} not found")

    # Check agent_sessions exist
    if not parent_session.agent_sessions:
        raise Exception(
            "No agent_sessions found. Call start_session first to initialize agents."
        )

    # Parse args
    topic = context.args.get("topic", "")
    choices = context.args.get("choices", [])
    voters = context.args.get("voters", [])
    reasoning_required = context.args.get("reasoning_required", False)

    if not topic:
        raise Exception("topic is required")
    if not choices or len(choices) < 2:
        raise Exception("At least 2 choices are required")
    if not voters:
        raise Exception("At least 1 voter is required")

    logger.info(
        f"[MODERATOR_VOTE] Starting vote: topic='{topic}', "
        f"choices={choices}, voters={voters}, reasoning_required={reasoning_required}"
    )

    # Build map of username -> (agent, agent_session_id)
    agent_map: Dict[str, tuple[Agent, ObjectId]] = {}
    for agent_id_str, agent_session_id in parent_session.agent_sessions.items():
        agent = Agent.from_mongo(ObjectId(agent_id_str))
        if agent:
            agent_map[agent.username.lower()] = (agent, agent_session_id)

    # Validate all voters exist in session
    valid_voters: List[tuple[Agent, ObjectId]] = []
    invalid_usernames = []

    for username in voters:
        username_lower = username.lower()
        if username_lower in agent_map:
            valid_voters.append(agent_map[username_lower])
        else:
            invalid_usernames.append(username)

    if invalid_usernames:
        valid_usernames = list(agent_map.keys())
        raise Exception(
            f"Invalid voters: {invalid_usernames}. Valid agents: {valid_usernames}"
        )

    # Collect votes in parallel batches
    all_votes: Dict[str, str] = {}  # username -> choice
    vote_details: List[Dict[str, str]] = []  # For logging

    async def collect_vote(
        agent: Agent, agent_session_id: ObjectId
    ) -> tuple[str, str, str]:
        """Collect a single vote from an agent using their workspace session."""
        try:
            username, choice, reasoning = await run_agent_vote_turn(
                parent_session=parent_session,
                agent_session_id=agent_session_id,
                actor=agent,
                topic=topic,
                choices=choices,
                reasoning_required=reasoning_required,
            )
            if reasoning:
                logger.info(
                    f"[MODERATOR_VOTE] {username} voted '{choice}': {reasoning[:50]}..."
                )
            else:
                logger.info(f"[MODERATOR_VOTE] {username} voted '{choice}'")
            return username, choice, reasoning
        except Exception as e:
            logger.error(
                f"[MODERATOR_VOTE] Error getting vote from {agent.username}: {e}"
            )
            return agent.username, "ERROR", str(e)[:100]

    # Process votes in batches of VOTE_BATCH_SIZE
    for i in range(0, len(valid_voters), VOTE_BATCH_SIZE):
        batch = valid_voters[i : i + VOTE_BATCH_SIZE]
        logger.info(
            f"[MODERATOR_VOTE] Processing batch {i // VOTE_BATCH_SIZE + 1}: "
            f"{[a.username for a, _ in batch]}"
        )

        # Run batch in parallel
        tasks = [collect_vote(agent, session_id) for agent, session_id in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[MODERATOR_VOTE] Batch vote error: {result}")
                continue
            username, choice, reasoning = result
            all_votes[username] = choice
            vote_detail = {
                "voter": username,
                "choice": choice,
            }
            if reasoning:
                vote_detail["reasoning"] = reasoning
            vote_details.append(vote_detail)

    # Tally votes
    tally: Dict[str, int] = {choice: 0 for choice in choices}
    for username, choice in all_votes.items():
        if choice in tally:
            tally[choice] += 1
        # Errors/ABSTAIN votes don't count toward valid choices

    logger.info(f"[MODERATOR_VOTE] Final tally: {tally}")

    # Build vote result content
    vote_result = {
        "topic": topic,
        "choices": choices,
        "voters": list(all_votes.keys()),
        "votes": vote_details,
        "tally": tally,
    }

    # Post MODERATOR_VOTE eden message to parent session
    eden_message = ChatMessage(
        session=[parent_session.id],
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content=json.dumps(vote_result),
        eden_message_data=EdenMessageData(message_type=EdenMessageType.MODERATOR_VOTE),
    )
    eden_message.save()

    logger.info(
        f"[MODERATOR_VOTE] Created MODERATOR_VOTE eden message {eden_message.id}"
    )

    return {
        "output": {
            "status": "success",
            "topic": topic,
            "tally": tally,
            "votes": vote_details,
            "total_votes": len(all_votes),
        }
    }
