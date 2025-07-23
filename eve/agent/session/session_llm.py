import logging
import os
import json
import litellm
from litellm import completion
from typing import Callable, List, AsyncGenerator, Optional

from eve.agent.thread import ChatMessage
from eve.agent.session.models import (
    LLMContext,
    LLMConfig,
    LLMContextMetadata,
    LLMResponse,
    LLMTraceMetadata,
    ToolCall,
)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


if os.getenv("LANGFUSE_TRACING_ENVIRONMENT"):
    litellm.success_callback = ["langfuse"]

supported_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-haiku-latest",
    "gemini-2.5-flash",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "gemini/gemini-2.5-flash-preview-05-20",
]


class ToolMetadataBuilder:
    def __init__(
        self,
        tool_name: str,
        litellm_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.litellm_session_id = litellm_session_id
        self.tool_name = tool_name
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id

    def __call__(self) -> LLMContextMetadata:
        return LLMContextMetadata(
            session_id=self.litellm_session_id,
            trace_name=f"TOOL_{self.tool_name}",
            generation_name=f"TOOL_{self.tool_name}",
            trace_metadata=LLMTraceMetadata(
                user_id=str(self.user_id),
                agent_id=str(self.agent_id),
                session_id=str(self.session_id),
            ),
        )


def validate_input(context: LLMContext) -> None:
    if context.config.model not in supported_models:
        raise ValueError(f"Model {context.config.model} is not supported")


def construct_observability_metadata(context: LLMContext):
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
        metadata["trace_metadata"] = context.metadata.trace_metadata.model_dump()
        metadata["trace_user_id"] = context.metadata.trace_metadata.user_id
    return metadata


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    if not context.tools:
        return None
    tools = [tool.openai_schema(exclude_hidden=True) for tool in context.tools.values()]
    
    # Fix for Gemini/Vertex AI: enum values must be strings and parameter type must be "string"
    if context.config.model and ("gemini" in context.config.model or "vertex" in context.config.model):
        for tool in tools:
            params = tool.get('function', {}).get('parameters', {}).get('properties', {})
            for param_name, param_def in params.items():
                if 'enum' in param_def:
                    # Convert all enum values to strings
                    param_def['enum'] = [str(val) for val in param_def['enum']]
                    # Ensure parameter type is "string" when using enum
                    param_def['type'] = 'string'
    
    return tools


async def async_run_tool_call(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
):
    tool = llm_context.tools[tool_call.tool]
    task = await tool.async_start_task(
        user_id=user_id,
        agent_id=agent_id,
        args=tool_call.args,
        mock=False,
        public=public,
        is_client_platform=is_client_platform,
    )

    result = await tool.async_wait(task)

    # Add task.cost and task.id to the result object
    if isinstance(result, dict):
        result["cost"] = getattr(task, "cost", None)
        result["task"] = getattr(task, "id", None)

    return result


def add_anthropic_cache_control(messages: List[dict]) -> List[dict]:
    """
    Add cache control for Anthropic models to optimize multi-turn conversations.

    - System messages are cached as static prefix
    - Second-to-last user message is cached as checkpoint
    - Final message is cached for continuation in follow-ups
    """
    # Add cache control to system message (static prefix)
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i]["cache_control"] = {"type": "ephemeral"}
            break

    # Find the last user message and second-to-last user message
    user_message_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_message_indices.append(i)

    # Mark second-to-last user message for caching (checkpoint)
    if len(user_message_indices) >= 2:
        messages[user_message_indices[-2]]["cache_control"] = {"type": "ephemeral"}

    # Mark the final message for continuing in followups
    if len(messages) > 0:
        messages[-1]["cache_control"] = {"type": "ephemeral"}

    return messages


def prepare_messages(
    messages: List[ChatMessage], model: Optional[str] = None
) -> List[dict]:
    messages = [schema for msg in messages for schema in msg.openai_schema()]

    # Add Anthropic cache control for models that support it
    if model and ("claude" in model or "anthropic" in model):
        messages = add_anthropic_cache_control(messages)

    return messages


async def async_prompt_litellm(
    context: LLMContext,
) -> LLMResponse:
    messages = prepare_messages(context.messages, context.config.model)
    tools = construct_tools(context)

    response = completion(
        model=context.config.model,
        messages=messages,
        metadata=construct_observability_metadata(context),
        tools=tools,
        response_format=context.config.response_format,
    )

    tool_calls = None
    if response.choices[0].message.tool_calls:
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                tool=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
                status="pending",
            )
            for tool_call in response.choices[0].message.tool_calls
        ]

    return LLMResponse(
        content=response.choices[0].message.content or "",  # content can't be None
        tool_calls=tool_calls,
        stop=response.choices[0].finish_reason,
        tokens_spent=response.usage.total_tokens,
    )


async def async_prompt_stream_litellm(
    context: LLMContext,
) -> AsyncGenerator[str, None]:
    response = await litellm.acompletion(
        model=context.config.model,
        messages=prepare_messages(context.messages, context.config.model),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
        stream=True,
        response_format=context.config.response_format,
    )
    async for part in response:
        yield part


DEFAULT_LLM_HANDLER = async_prompt_litellm
DEFAULT_LLM_STREAM_HANDLER = async_prompt_stream_litellm


async def async_prompt(
    context: LLMContext,
    handler: Optional[Callable[[LLMContext], str]] = DEFAULT_LLM_HANDLER,
) -> LLMResponse:
    validate_input(context)
    response = await handler(context)
    return response


async def async_prompt_stream(
    context: LLMContext,
    handler: Optional[
        Callable[[LLMContext, LLMConfig], AsyncGenerator[str, None]]
    ] = DEFAULT_LLM_STREAM_HANDLER,
) -> AsyncGenerator[str, None]:
    validate_input(context)
    async for chunk in handler(context):
        yield chunk
