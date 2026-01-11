"""Handler for moderator finish_session tool.

This tool ends the multi-agent session with a summary and outcome.
"""

from typing import Any, Dict, Optional

from bson import ObjectId
from loguru import logger
from pydantic import BaseModel

from eve.agent.session.models import (
    ChatMessage,
    EdenMessageData,
    EdenMessageType,
    Session,
)
from eve.tool import ToolContext


class ModeratorFinishResponse(BaseModel):
    """Finish session response data."""

    summary: str
    outcome: Optional[str] = None


async def handler(context: ToolContext) -> Dict[str, Any]:
    """End the multi-agent session.

    Posts a MODERATOR_FINISH eden message and pauses the session.

    Args:
        context: ToolContext containing:
            - args.summary: Summary of what happened
            - args.outcome: Optional resolution/final state
            - session: The moderator_session ID (which has a parent_session)

    Returns:
        Dict with status confirmation

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

    # Parse args
    summary = context.args.get("summary", "")
    outcome = context.args.get("outcome")

    if not summary:
        raise Exception("summary is required")

    logger.info(
        f"[MODERATOR_FINISH] Finishing session {parent_session.id}: {summary[:100]}..."
    )

    # Build finish response
    finish_response = ModeratorFinishResponse(
        summary=summary,
        outcome=outcome,
    )

    # Create MODERATOR_FINISH eden message in parent session
    eden_content = finish_response.model_dump_json()
    eden_message = ChatMessage(
        session=[parent_session.id],
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content=eden_content,
        eden_message_data=EdenMessageData(
            message_type=EdenMessageType.MODERATOR_FINISH
        ),
    )
    eden_message.save()

    logger.info(
        f"[MODERATOR_FINISH] Created MODERATOR_FINISH eden message {eden_message.id}"
    )

    # Mark the parent session as finished
    parent_session.update(status="finished")
    logger.info(
        f"[MODERATOR_FINISH] Session {parent_session.id} status set to 'finished'"
    )

    return {
        "output": {
            "status": "success",
            "message": "Session finished",
            "summary": summary,
            "outcome": outcome,
        }
    }
