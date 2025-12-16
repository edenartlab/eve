"""
Hook for handling reactions to verdelis_plant_seed tool calls.

When a user reacts to a seed creation, this hook posts a message to the session
asking Verdelis to draft a storyboard from that seed using verdelis_draft_storyboard.
"""

import logging

from bson import ObjectId

from eve.agent.session.models import ChatMessage
from eve.tool import Tool
from eve.tools.verdelis.verdelis_seed.handler import VerdelisSeed

logger = logging.getLogger(__name__)


async def _run_hook(message_id: str, tool_call_id: str, reaction: str, user_id: str):
    """Async implementation of the hook."""
    # Load the message to get session and tool result
    message = ChatMessage.from_mongo(ObjectId(message_id))
    if not message:
        logger.warning(f"[verdelis_plant_seed hook] Message not found: {message_id}")
        return

    session_id = str(message.session[0]) if message.session else None

    # Find the tool call and extract artifact info from result
    artifact_id = None
    artifact_ids = []
    for tc in message.tool_calls or []:
        if tc.id == tool_call_id:
            if tc.result:
                # Result is a list of dicts, first item contains the output
                for result_item in tc.result:
                    if isinstance(result_item, dict):
                        output = result_item.get("output", {})
                        if isinstance(output, dict):
                            artifact_id = output.get("artifact_id")
                            artifact_ids = output.get("artifact_ids", [])
            break

    logger.info(
        f"[verdelis_plant_seed hook] Reaction received!\n"
        f"  reaction: {reaction}\n"
        f"  message_id: {message_id}\n"
        f"  session_id: {session_id}\n"
        f"  artifact_id: {artifact_id}\n"
        f"  artifact_ids: {artifact_ids}\n"
        f"  user_id: {user_id}"
    )

    # If we have an artifact_id and session_id, post a message to draft a storyboard
    if artifact_id and session_id:
        try:
            # Load the seed to get its title
            seed = VerdelisSeed.from_mongo(ObjectId(artifact_id))
            if not seed:
                logger.error(
                    f"[verdelis_plant_seed hook] Seed not found: {artifact_id}"
                )
                return

            # Load session_post tool to post a message
            session_post = Tool.load("session_post")

            # Get the session to find the agent
            from eve.agent.session.models import Session

            session = Session.from_mongo(ObjectId(session_id))
            if not session or not session.agents:
                logger.error("[verdelis_plant_seed hook] Session or agents not found")
                return

            agent_id = str(session.agents[0])
            owner_id = str(session.owner) if session.owner else user_id

            # Post to the original session asking to run verdelis_draft_storyboard
            content = f"""
The user selected the following Seed:

<SelectedSeed>
ID: {artifact_id}
Title: {seed.title}
Logline: {seed.logline}
Agents: {seed.agents}
</SelectedSeed>

Their comments (feedback, suggestions, etc.): {reaction}

Run verdelis_draft_storyboard to draft a storyboard based on this seed, taking into account the user's comments.
"""

            # Post to the ORIGINAL session with verdelis_draft_storyboard in extra_tools
            result = await session_post.async_run(
                {
                    "role": "user",
                    "user_id": owner_id,
                    "agent_id": agent_id,
                    "session": session_id,  # Post to the original session
                    "content": content,
                    "attachments": seed.images,  # Seed images as references
                    "prompt": True,
                    "async": True,  # Returns immediately
                    "extra_tools": ["verdelis_draft_storyboard"],
                }
            )

            logger.info(
                f"[verdelis_plant_seed hook] Posted to session {session_id}. "
                f"Result: {result}"
            )

        except Exception as e:
            logger.error(f"[verdelis_plant_seed hook] Error posting message: {e}")

    return {
        "status": "hook_executed",
        "tool": "verdelis_plant_seed",
        "reaction": reaction,
        "artifact_id": artifact_id,
        "artifact_ids": artifact_ids,
        "session_id": session_id,
    }


def hook(message_id: str, tool_call_id: str, reaction: str, user_id: str):
    """
    Handle a reaction to a verdelis_plant_seed tool call.

    Posts a message to the session asking Verdelis to draft a storyboard from the seed.
    """
    # Return the coroutine - the handler will await it if needed
    return _run_hook(message_id, tool_call_id, reaction, user_id)
