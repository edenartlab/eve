import copy
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm import litellm
from loguru import logger

from eve.agent.llm.constants import TEST_MODE_TEXT_STRING, TEST_MODE_TOOL_STRING
from eve.agent.session.models import ChatMessage, LLMContext, ToolCall


def _extract_image_urls_from_content(content: List[Dict[str, Any]]) -> List[str]:
    """Extract cloudfront image URLs from text blocks in message content.

    Looks for URLs in the <attachments> section like:
    * https://dtut5r9j4w7j4.cloudfront.net/xxx.jpg
    """
    urls = []
    for block in content:
        if block.get("type") == "text":
            text = block.get("text", "")
            # Match cloudfront URLs in attachments section
            matches = re.findall(
                r"https://[a-z0-9]+\.cloudfront\.net/[^\s\n\)]+",
                text,
            )
            urls.extend(matches)
    return urls


def truncate_base64_in_payload(
    payload: Dict[str, Any], max_length: int = 30
) -> Dict[str, Any]:
    """Recursively truncate base64 image data in request payloads for storage.

    Handles both Anthropic and OpenAI image formats:
    - Anthropic: {"type": "image", "source": {"type": "base64", "data": "..."}}
    - OpenAI: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
    - Google: {"inline_data": {"mime_type": "...", "data": "..."}}

    Also adds a "url" field to truncated image sources with the original cloudfront URL
    extracted from the attachments text block.
    """
    result = copy.deepcopy(payload)

    # Process messages to extract URLs and add them to image blocks
    messages = result.get("messages", [])
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        # Extract URLs from text blocks in this message
        image_urls = _extract_image_urls_from_content(content)
        url_index = 0

        # Process each block in the content
        for block in content:
            # Handle Anthropic format
            if block.get("type") == "image" and "source" in block:
                source = block["source"]
                if source.get("type") == "base64" and "data" in source:
                    data = source["data"]
                    if isinstance(data, str) and len(data) > max_length:
                        source["data"] = data[:max_length] + "..."
                        # Add URL if available
                        if url_index < len(image_urls):
                            source["url"] = image_urls[url_index]
                            url_index += 1

            # Handle OpenAI format
            elif block.get("type") == "image_url" and "image_url" in block:
                image_url = block["image_url"]
                if "url" in image_url:
                    url = image_url["url"]
                    if isinstance(url, str) and ";base64," in url:
                        prefix, b64_data = url.split(";base64,", 1)
                        if len(b64_data) > max_length:
                            image_url["url"] = (
                                f"{prefix};base64,{b64_data[:max_length]}..."
                            )
                            # Add original URL if available
                            if url_index < len(image_urls):
                                image_url["original_url"] = image_urls[url_index]
                                url_index += 1

            # Handle tool_result content (Anthropic)
            elif block.get("type") == "tool_result" and isinstance(
                block.get("content"), list
            ):
                tool_content = block["content"]
                tool_image_urls = _extract_image_urls_from_content(tool_content)
                tool_url_index = 0
                for tool_block in tool_content:
                    if tool_block.get("type") == "image" and "source" in tool_block:
                        source = tool_block["source"]
                        if source.get("type") == "base64" and "data" in source:
                            data = source["data"]
                            if isinstance(data, str) and len(data) > max_length:
                                source["data"] = data[:max_length] + "..."
                                if tool_url_index < len(tool_image_urls):
                                    source["url"] = tool_image_urls[tool_url_index]
                                    tool_url_index += 1

    # Handle Google format in contents
    contents = result.get("contents", [])
    for content_item in contents:
        parts = content_item.get("parts", [])
        for part in parts:
            if "inline_data" in part:
                inline_data = part["inline_data"]
                if "data" in inline_data:
                    data = inline_data["data"]
                    if isinstance(data, str) and len(data) > max_length:
                        inline_data["data"] = data[:max_length] + "..."

    return result


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
            entry["tool_calls"] = [
                tc.model_dump(exclude_none=True) for tc in message.tool_calls
            ]  # type: ignore[arg-type]
        serialized.append(entry)
    return serialized


def build_langfuse_prompt(context: LLMContext) -> Dict[str, Any]:
    """Construct a Langfuse-compatible prompt structure from the LLM context."""
    return {
        "type": "chat",
        "messages": serialize_context_messages(context),
    }
