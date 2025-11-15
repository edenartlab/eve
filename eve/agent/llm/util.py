import os
from typing import Any, Dict, List, Optional, Tuple

from litellm import litellm
from loguru import logger

from eve.agent.llm.constants import TEST_MODE_TEXT_STRING, TEST_MODE_TOOL_STRING
from eve.agent.session.models import ChatMessage, LLMContext, ToolCall


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


def calculate_cost_usd(
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Tuple[float, float, float]:
    """Return (prompt_cost, completion_cost, total_cost)."""
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        logger.warning(f"Failed to calculate cost for model {model}: {exc}")
        return 0.0, 0.0, 0.0

    total = (prompt_cost or 0.0) + (completion_cost or 0.0)
    return prompt_cost or 0.0, completion_cost or 0.0, total


def serialize_context_messages(context: LLMContext) -> List[Dict[str, Optional[str]]]:
    """Return a simplified representation of the chat history for observability."""
    serialized: List[Dict[str, Optional[str]]] = []
    for message in context.messages:
        entry: Dict[str, Optional[str]] = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        if message.tool_calls:
            entry["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]  # type: ignore[arg-type]
        serialized.append(entry)
    return serialized


def build_langfuse_prompt(context: LLMContext) -> Dict[str, Any]:
    """Construct a Langfuse-compatible prompt structure from the LLM context."""
    return {
        "type": "chat",
        "messages": serialize_context_messages(context),
    }
