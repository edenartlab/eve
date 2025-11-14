import os
from typing import Optional

from eve.agent.llm.constants import TEST_MODE_TEXT_STRING, TEST_MODE_TOOL_STRING
from eve.agent.session.models import (
    ChatMessage,
    LLMContext,
    ToolCall,
)


def validate_input(context: LLMContext) -> None:
    """Placeholder for future schema validation."""
    return None


def is_fake_llm_mode() -> bool:
    """Feature flag toggle for fake LLM responses."""
    return os.getenv("FF_SESSION_FAKE_LLM", "").lower() in {"1", "true", "yes", "on"}


def is_test_mode_prompt(prompt_text: Optional[str]) -> bool:
    """Check if the raw prompt text should force fake/test handling."""
    if not prompt_text:
        return False

    cleaned = prompt_text.strip()
    if cleaned in {TEST_MODE_TEXT_STRING, TEST_MODE_TOOL_STRING}:
        return True

    return TEST_MODE_TEXT_STRING in cleaned or TEST_MODE_TOOL_STRING in cleaned


def get_last_user_prompt(context: LLMContext) -> Optional[str]:
    """Return the content of the most recent user message."""
    if not context.messages:
        return None

    for message in reversed(context.messages):
        if message.role == "user" and message.content:
            return message.content.strip()
    return None


def get_last_assistant_message(context: LLMContext) -> Optional[ChatMessage]:
    """Return the most recent assistant message."""
    if not context.messages:
        return None

    for message in reversed(context.messages):
        if message.role == "assistant":
            return message
    return None


def get_last_completed_tool_call(context: LLMContext) -> Optional[ToolCall]:
    """Return the most recent completed tool call from assistant messages."""
    if not context.messages:
        return None

    for message in reversed(context.messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        for tool_call in reversed(message.tool_calls):
            if tool_call.status == "completed" and tool_call.result:
                return tool_call
    return None


def should_force_fake_response(context: LLMContext) -> bool:
    """Determine whether the request should bypass providers and use fake handlers."""
    prompt_text = get_last_user_prompt(context)
    if prompt_text and is_test_mode_prompt(prompt_text):
        return True
    return is_fake_llm_mode()
