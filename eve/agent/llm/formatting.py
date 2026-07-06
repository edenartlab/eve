from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from eve.agent.session.models import ChatMessage, LLMContext

# Marker rendered into the system prompt (see system_template.py) at the boundary
# between the static, per-agent prefix (identity, persona, rules, tool guidance)
# and the volatile tail (memory, session context, current date, trigger task).
# The Anthropic provider splits on it to place prompt-cache breakpoints; other
# providers strip it. It is a harmless XML comment if it ever leaks to a model.
SYSTEM_CACHE_BREAKPOINT = "<!--eve:cache-breakpoint-->"


def split_system_for_cache(
    system: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Split a rendered system prompt into (static_prefix, volatile_tail).

    Returns (whole, None) when no breakpoint marker is present.
    """
    if not system:
        return system, None
    if SYSTEM_CACHE_BREAKPOINT not in system:
        return system, None
    static, _, volatile = system.partition(SYSTEM_CACHE_BREAKPOINT)
    return (static.strip() or None), (volatile.strip() or None)


def strip_cache_breakpoint(system: Optional[str]) -> Optional[str]:
    """Remove the cache-breakpoint marker (for non-Anthropic providers)."""
    if not system:
        return system
    return system.replace(SYSTEM_CACHE_BREAKPOINT, "").strip() or None


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

    tools = context.tools or {}
    if not tools:
        return None
    if isinstance(tools, dict):
        iter_tools = tools.values()
    else:
        iter_tools = tools

    tool_schemas = [tool.anthropic_schema(exclude_hidden=True) for tool in iter_tools]

    # Stable order so the tool block is a byte-identical, cacheable prefix across
    # turns (tools render before system, so their order affects the whole cache).
    tool_schemas.sort(key=lambda t: t.get("name", ""))

    return tool_schemas


def construct_gemini_tools(context: LLMContext) -> Optional[List[dict]]:
    """Build Gemini function declarations from an LLMContext.

    Gemini uses a different format than OpenAI:
    - Tools are wrapped in a Tool object with function_declarations
    - Parameters use "object" type with properties
    - Enum values must be strings
    - Non-standard JSON Schema properties are rejected
    """
    tools = context.tools or {}
    if not tools:
        return None
    if isinstance(tools, dict):
        iter_tools = tools.values()
    else:
        iter_tools = tools

    function_declarations = []
    for tool in iter_tools:
        # Use gemini_schema which strips non-standard properties
        gemini_schema = tool.gemini_schema(exclude_hidden=True)
        if not gemini_schema:
            continue

        parameters = gemini_schema.get("parameters", {})

        # Ensure enum values are strings and type is "string" for enum params
        if "properties" in parameters:
            for param_def in parameters["properties"].values():
                if "enum" in param_def:
                    param_def["enum"] = [str(val) for val in param_def["enum"]]
                    param_def["type"] = "string"

        function_declarations.append(gemini_schema)

    return function_declarations if function_declarations else None


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
