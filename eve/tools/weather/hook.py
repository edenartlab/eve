"""
Example hook for handling reactions to tool calls.

This hook is optional - if a tool doesn't have a hook.py file,
no hook will be triggered when users react to that tool's outputs.

The hook function receives:
- message_id: The ID of the message containing the tool call
- tool_call_id: The ID of the specific tool call that was reacted to
- reaction: The reaction emoji or key (e.g., "thumbs_up", "fire")
- user_id: The ID of the user who reacted
"""

import logging

logger = logging.getLogger(__name__)


def hook(message_id: str, tool_call_id: str, reaction: str, user_id: str):
    """
    Handle a reaction to a weather tool call.

    This is a simple example that just logs the reaction info.
    You can extend this to do things like:
    - Update analytics/metrics
    - Trigger follow-up actions based on positive/negative feedback
    - Store user preferences
    - Send notifications
    """
    logger.info(
        f"[weather hook] Reaction received: "
        f"message_id={message_id}, "
        f"tool_call_id={tool_call_id}, "
        f"reaction={reaction}, "
        f"user_id={user_id}"
    )

    # Example: You could add custom logic here based on the reaction
    # if reaction in ["thumbs_up", "heart", "fire"]:
    #     logger.info("Positive feedback received!")
    # elif reaction in ["thumbs_down"]:
    #     logger.info("Negative feedback received - consider improving the response")

    return {"status": "hook_executed", "tool": "weather"}
