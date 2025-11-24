from __future__ import annotations

import json
from typing import Dict, List, Optional

from eve.agent.session.models import ChatMessage, LLMContext


def add_anthropic_cache_control(messages: List[dict]) -> List[dict]:
    """
    Apply Anthropic cache control directives to the first system/user messages
    to optimize multi-turn conversations.
    """
    if not messages:
        return messages

    # Add cache control to system message (static prefix)
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i]["cache_control"] = {"type": "ephemeral"}
            break

    # Mark the final message for continuing in followups
    messages[-1]["cache_control"] = {"type": "ephemeral"}
    return messages


def prepare_messages(
    messages: List[ChatMessage],
    model: Optional[str] = None,
    include_thoughts: bool = False,
) -> List[dict]:
    """Convert internal ChatMessage objects into OpenAI-compatible schemas."""
    prepared = [
        schema
        for msg in messages
        for schema in msg.openai_schema(include_thoughts=include_thoughts)
    ]

    if model and ("claude" in model or "anthropic" in model):
        prepared = add_anthropic_cache_control(prepared)

    return prepared


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    """Build provider-ready tool definitions from an LLMContext."""
    tools = context.tools or {}
    if isinstance(tools, dict):
        iter_tools = tools.values()
    else:
        iter_tools = tools

    tool_schemas = [tool.openai_schema(exclude_hidden=True) for tool in iter_tools]

    # Gemini/Vertex: enum values must be strings and parameter type must be "string"
    if context.config.model and (
        "gemini" in context.config.model or "vertex" in context.config.model
    ):
        for tool in tool_schemas:
            params = (
                tool.get("function", {}).get("parameters", {}).get("properties", {})
            )
            for param_def in params.values():
                if "enum" in param_def:
                    param_def["enum"] = [str(val) for val in param_def["enum"]]
                    param_def["type"] = "string"

    return tool_schemas


def construct_anthropic_tools(context: LLMContext) -> Optional[List[dict]]:
    """Build Anthropic tool definitions."""
    from loguru import logger

    tools = context.tools or {}
    if not tools:
        return None
    if isinstance(tools, dict):
        iter_tools = tools.values()
    else:
        iter_tools = tools

    tool_schemas = [tool.anthropic_schema(exclude_hidden=True) for tool in iter_tools]

    # Log tool names being sent to Anthropic
    tool_names = [schema.get("name") for schema in tool_schemas if "name" in schema]
    logger.info(f"Sending {len(tool_names)} tools to Anthropic: {tool_names}")

    return tool_schemas


def construct_observability_metadata(context: LLMContext) -> Dict[str, str]:
    """Flatten LLM metadata for observability."""
    if not context.metadata:
        return {}
    metadata = {
        "session_id": context.metadata.session_id,
        "trace_id": context.metadata.trace_id,
        "trace_name": context.metadata.trace_name,
        "generation_name": context.metadata.generation_name,
        "generation_id": context.metadata.generation_id,
    }
    if context.metadata.trace_metadata:
        trace_payload = context.metadata.trace_metadata.model_dump()
        metadata["trace_metadata"] = json.dumps(trace_payload)
        metadata["trace_user_id"] = context.metadata.trace_metadata.user_id
    return {
        k: (json.dumps(v) if isinstance(v, dict) else v)
        for k, v in metadata.items()
        if v is not None
    }
