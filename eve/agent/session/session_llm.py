import logging
import os
import time
import uuid
import json
import litellm
from litellm import acompletion
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
    "gemini-2.5-flash",
    "claude-3-haiku-20240307",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "gemini/gemini-2.5-flash-preview-05-20",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07"
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
    tools = context.tools or {}

    tools = [
        tool.openai_schema(exclude_hidden=True) 
        for tool in tools.values()
    ]

    # Gemini/Vertex: enum values must be strings and parameter type must be "string"
    if "gemini" in context.config.model or "vertex" in context.config.model:
        for tool in tools:
            params = (
                tool.get("function", {}).get("parameters", {}).get("properties", {})
            )
            for param_def in params.values():
                if "enum" in param_def:
                    param_def["enum"] = [str(val) for val in param_def["enum"]]
                    param_def["type"] = "string"

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

    completion_kwargs = {
        "model": context.config.model,
        "messages": messages,
        "metadata": construct_observability_metadata(context),
        "tools": tools,
        "response_format": context.config.response_format,
        "fallbacks": context.config.fallback_models,
        "drop_params": True,
        "num_retries": 2,
        "timeout": 300,
        "context_window_fallback_dict": {
            # promote to larger context sibling if overflow detected
            "gpt-4o-mini": "gpt-4o",
            "claude-3-5-haiku-20241022": "claude-3-5-sonnet-20241022",
        },
    }
    

    # add web search options for Anthropic models
    # todo: does this fail in fallback models?
    if "claude" in context.config.model:
        completion_kwargs["web_search_options"] = {
            "search_context_size": "medium"
        }
    
    # Use finalized reasoning_effort from config if available
    if context.config.reasoning_effort:
        completion_kwargs["reasoning_effort"] = context.config.reasoning_effort
    
    logging.info(f"Attempting completion with model: {context.config.model}, fallbacks: {context.config.fallback_models}")
    
    try:
        t0 = time.time()
        response = await acompletion(**completion_kwargs)
        t1 = time.time()
        
        actual_model = getattr(response, "model", context.config.model)
        
        if actual_model != context.config.model and context.config.fallback_models:
            logging.info("Response received from fallback model: %s", actual_model)
            
    except Exception as e:
        logging.error(f"All models failed. Error: {str(e)}")
        raise

    tool_calls = []
    
    # add web search as a tool call
    psf = getattr(response.choices[0].message, "provider_specific_fields", None)
    if psf:
        citations = psf.get("citations") or []
        sources = []
        for citation_block in citations:
            for citation in citation_block:
                source = {
                    "title": citation.get('title'),
                    "url": citation.get('url'),
                }
                if not source in sources:  # avoid duplicates
                    sources.append(source)
        if sources:
            tool_calls.append(
                ToolCall(
                    id=f"toolu_{uuid.uuid4()}",
                    tool="web_search",
                    args={},
                    result=sources,
                    status="completed",
                )
            )

    # add regular tool calls
    if response.choices[0].message.tool_calls:
        tool_calls.extend([
            ToolCall(
                id=tool_call.id,
                tool=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
                status="pending",
            )
            for tool_call in response.choices[0].message.tool_calls
        ])

    # Extract thinking blocks if present
    # todo: look at reasoning_content
    thought = None
    if hasattr(response.choices[0].message, 'thinking_blocks') and response.choices[0].message.thinking_blocks and len(response.choices[0].message.thinking_blocks) > 0:
        thought = response.choices[0].message.thinking_blocks
        seconds = t1 - t0
        if seconds < 60:
            thought[0]["title"] = f"Thought for {seconds:.0f} seconds"
        else:
            thought[0]["title"] = f"Thought for {round(seconds/60)} minutes"

    return LLMResponse(
        content=response.choices[0].message.content or "",
        tool_calls=tool_calls or None,
        stop=response.choices[0].finish_reason,
        tokens_spent=response.usage.total_tokens,
        thought=thought,
    )


async def async_prompt_stream_litellm(
    context: LLMContext,
) -> AsyncGenerator[str, None]:
    # Todo: if we use stream again, add web search here like we do in async_prompt_litellm
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
