import logging

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

import json
from litellm import completion
import litellm
from typing import Callable, List, AsyncGenerator, Optional

from eve.agent.session.models import LLMContext, LLMConfig, LLMResponse, ToolCall


litellm.success_callback = ["langfuse"]

supported_models = ["gpt-4o-mini", "gpt-4o"]


def validate_input(context: LLMContext) -> None:
    if context.config.model not in supported_models:
        raise ValueError(f"Model {context.config.model} is not supported")


def construct_observability_metadata(context: LLMContext):
    if not context.metadata:
        return {}
    return {
        "trace_name": context.metadata.trace_name,
        "generation_name": context.metadata.generation_name,
        "trace_metadata": context.metadata.trace_metadata.model_dump(),
    }


def construct_messages(context: LLMContext) -> List[dict]:
    return [msg.openai_schema() for msg in context.messages]


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    if not context.tools:
        return None
    return [tool.openai_schema(exclude_hidden=True) for tool in context.tools.values()]


async def async_prompt_litellm(
    context: LLMContext,
) -> LLMResponse:
    messages = construct_messages(context)
    print(f"***debug*** MESSAGES: {messages}")
    response = completion(
        model=context.config.model,
        messages=messages,
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
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
    )


async def async_prompt_stream_litellm(
    context: LLMContext,
) -> AsyncGenerator[str, None]:
    response = completion(
        model=context.config.model,
        messages=construct_messages(context),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
        stream=True,
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
    return await handler(context)


async def async_prompt_stream(
    context: LLMContext,
    handler: Optional[
        Callable[[LLMContext, LLMConfig], AsyncGenerator[str, None]]
    ] = DEFAULT_LLM_STREAM_HANDLER,
) -> AsyncGenerator[str, None]:
    validate_input(context)
    async for chunk in handler(context):
        yield chunk
