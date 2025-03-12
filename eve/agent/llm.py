import os
import json
import asyncio
import anthropic
from enum import Enum
from typing import Optional, Dict, List, Union, Literal, Tuple, AsyncGenerator
from pydantic import BaseModel
from instructor.function_calls import openai_schema
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import openai

from ..tool import Tool
from ..eden_utils import dump_json
from .thread import UserMessage, AssistantMessage, ToolCall


MODELS = [
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
]

DEFAULT_MODEL = "claude-3-5-haiku-latest"


class UpdateType(str, Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    END_PROMPT = "end_prompt"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_STOP = "assistant_stop"
    TOOL_CALL = "tool_call"


def calculate_anthropic_model_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prompt_cache_write_tokens: int = 0,
    cached_tokens: int = 0,
) -> Dict[str, float]:
    """
    Calculate the cost of a model call based on token usage.

    Args:
        model: The model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached tokens (for prompt caching)

    Returns:
        Dictionary with cost details
    """

    # Claude 3.7 Sonnet
    if model == "claude-3-7-sonnet-latest":
        # Prices per million tokens ($X/MTok)
        input_price = 3.0 / 1_000_000  # $3/MTok
        prompt_cache_write_price = 3.75 / 1_000_000  # $0.30/MTok for cached tokens
        output_price = 15.0 / 1_000_000  # $15/MTok
        cached_price = 0.30 / 1_000_000  # $0.30/MTok for cached tokens

        # Calculate costs
        input_cost = input_tokens * input_price
        prompt_cache_write_cost = prompt_cache_write_tokens * prompt_cache_write_price
        output_cost = output_tokens * output_price
        cached_cost = cached_tokens * cached_price
        total_cost = input_cost + prompt_cache_write_cost + output_cost + cached_cost

        return {
            "input": input_cost + prompt_cache_write_cost,
            "output": output_cost,
            "total": total_cost,
        }
    
    # Claude 3.5 Haiku
    elif model == "claude-3-5-haiku-latest":
        # Prices per million tokens ($X/MTok)
        input_price = 0.8 / 1_000_000  # $3/MTok
        output_price = 4.0 / 1_000_000  # $15/MTok
        prompt_cache_write_price = 1.0 / 1_000_000  # $0.30/MTok for cached tokens
        cached_price = 0.08 / 1_000_000  # $0.30/MTok for cached tokens

        # Calculate costs
        input_cost = input_tokens * input_price
        prompt_cache_write_cost = prompt_cache_write_tokens * prompt_cache_write_price
        output_cost = output_tokens * output_price
        cached_cost = cached_tokens * cached_price
        total_cost = input_cost + prompt_cache_write_cost + output_cost + cached_cost

        return {
            "input": input_cost + prompt_cache_write_cost,
            "output": output_cost,
            "total": total_cost,
        }

    # Default fallback pricing
    return {"input": 0, "output": 0, "cache_read_input_tokens": 0, "total": 0}


@observe(as_type="generation")
async def async_anthropic_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: Literal[tuple(MODELS)] = "claude-3-5-haiku-latest",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
):
    anthropic_client = anthropic.AsyncAnthropic()

    prompt = {
        "model": model,
        "max_tokens": 8192,
        "messages": [item for msg in messages for item in msg.anthropic_schema()],
        "system": [
            {
                "type": "text",
                "text": system_message,
                # "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    # efficient tool calls feature for claude 3.7
    # if "claude-3-7" in model:
    #     prompt["betas"] = ["token-efficient-tools-2025-02-19"]

    # add tools / structure output
    if tools or response_model:
        tool_schemas = [
            t.anthropic_schema(exclude_hidden=True) for t in (tools or {}).values()
        ]
        if response_model:
            tool_schemas.append(openai_schema(response_model).anthropic_schema)
            prompt["tool_choice"] = {"type": "tool", "name": response_model.__name__}

        # cache tools
        tool_schemas[-1]["cache_control"] = {"type": "ephemeral"}

        prompt["tools"] = tool_schemas

    import json
    import time

    start_time = time.time()

    # call Anthropic
    response = await anthropic_client.messages.create(**prompt)

    
    # print("-----------PROMPT---------------------")
    # print(json.dumps(prompt["tools"], indent=2))
    # print("--------------------------------")

    print("-----------RESPONSE USAGE---------------------")
    print(f"Time taken: {time.time() - start_time} seconds")
    print(response.usage)
    print("--------------------------------")

    # Get token usage
    input_tokens = response.usage.input_tokens + getattr(
        response.usage, "cache_creation_input_tokens", 0
    )
    output_tokens = response.usage.output_tokens
    cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0)

    # Calculate cost
    cost = calculate_anthropic_model_cost(model, input_tokens, output_tokens, cached_tokens)

    # Update Langfuse observation with usage and cost details
    langfuse_context.update_current_observation(
        usage_details={
            "input": input_tokens + cached_tokens,
            "output": output_tokens,
        },
        cost_details=cost,
    )

    if response_model:
        return response_model(**response.content[0].input)
    else:
        content = ". ".join(
            [r.text for r in response.content if r.type == "text" and r.text]
        )
        tool_calls = [
            ToolCall.from_anthropic(r) for r in response.content if r.type == "tool_use"
        ]
        stop = response.stop_reason == "end_turn"
        return content, tool_calls, stop


@observe(as_type="generation")
async def async_anthropic_prompt_stream(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: Literal[tuple(MODELS)] = "claude-3-5-haiku-latest",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
) -> AsyncGenerator[Tuple[UpdateType, str], None]:
    """Yields partial tokens (ASSISTANT_TOKEN, partial_text) for streaming."""
    anthropic_client = anthropic.AsyncAnthropic()
    prompt = {
        "model": model,
        "max_tokens": 8192,
        "messages": [item for msg in messages for item in msg.anthropic_schema()],
        "system": system_message,
    }

    if tools or response_model:
        tool_schemas = [
            t.anthropic_schema(exclude_hidden=True) for t in (tools or {}).values()
        ]
        if response_model:
            tool_schemas.append(openai_schema(response_model).anthropic_schema)
            prompt["tool_choice"] = {"type": "tool", "name": response_model.__name__}
        prompt["tools"] = tool_schemas

    tool_calls = []

    async with anthropic_client.messages.stream(**prompt) as stream:
        async for chunk in stream:
            # Handle text deltas
            if (
                chunk.type == "content_block_delta"
                and chunk.delta
                and hasattr(chunk.delta, "text")
                and chunk.delta.text
            ):
                yield (UpdateType.ASSISTANT_TOKEN, chunk.delta.text)

            # Handle tool use
            elif chunk.type == "content_block_stop" and hasattr(chunk, "content_block"):
                if chunk.content_block.type == "tool_use":
                    tool_calls.append(ToolCall.from_anthropic(chunk.content_block))

            # Stop reason
            elif chunk.type == "message_delta" and hasattr(chunk.delta, "stop_reason"):
                yield (UpdateType.ASSISTANT_STOP, chunk.delta.stop_reason)

    # Return any accumulated tool calls at the end
    if tool_calls:
        for tool_call in tool_calls:
            yield (UpdateType.TOOL_CALL, tool_call)


@observe()
async def async_openai_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: Literal[tuple(MODELS)] = "gpt-4o-mini",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY env is not set")

    messages_json = [item for msg in messages for item in msg.openai_schema()]
    if system_message:
        messages_json = [{"role": "system", "content": system_message}] + messages_json

    openai_client = openai.AsyncOpenAI()

    if response_model:
        response = await openai_client.beta.chat.completions.parse(
            model=model, messages=messages_json, response_format=response_model
        )
        return response.choices[0].message.parsed

    else:
        tools = (
            [t.openai_schema(exclude_hidden=True) for t in tools.values()]
            if tools
            else None
        )
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages_json, tools=tools
        )
        response = response.choices[0]
        content = response.message.content or ""
        tool_calls = [
            ToolCall.from_openai(t) for t in response.message.tool_calls or []
        ]
        stop = response.finish_reason == "stop"

        return content, tool_calls, stop


async def async_openai_prompt_stream(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: Literal[tuple(MODELS)] = "gpt-4o-mini",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
) -> AsyncGenerator[Tuple[UpdateType, str], None]:
    """Yields partial tokens (ASSISTANT_TOKEN, partial_text) for streaming."""

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY env is not set")

    messages_json = [item for msg in messages for item in msg.openai_schema()]
    if system_message:
        messages_json = [{"role": "system", "content": system_message}] + messages_json

    openai_client = openai.AsyncOpenAI()
    tools_schema = (
        [t.openai_schema(exclude_hidden=True) for t in tools.values()]
        if tools
        else None
    )

    if response_model:
        # Response models not supported in streaming mode for OpenAI
        raise NotImplementedError(
            "Response models not supported in streaming mode for OpenAI"
        )

    stream = await openai_client.chat.completions.create(
        model=model, messages=messages_json, tools=tools_schema, stream=True
    )

    tool_calls = []

    async for chunk in stream:
        delta = chunk.choices[0].delta

        # Handle text content
        if delta.content:
            yield (UpdateType.ASSISTANT_TOKEN, delta.content)

        # Handle tool calls
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.index is not None:
                    # Ensure we have a list long enough
                    while len(tool_calls) <= tool_call.index:
                        tool_calls.append(None)

                    if tool_calls[tool_call.index] is None:
                        tool_calls[tool_call.index] = ToolCall(
                            tool=tool_call.function.name, args={}
                        )

                    if tool_call.function.arguments:
                        current_args = tool_calls[tool_call.index].args
                        # Merge new arguments with existing ones
                        try:
                            new_args = json.loads(tool_call.function.arguments)
                            current_args.update(new_args)
                        except json.JSONDecodeError:
                            pass

        # Handle finish reason
        if chunk.choices[0].finish_reason:
            yield (UpdateType.ASSISTANT_STOP, chunk.choices[0].finish_reason)

    # Yield any accumulated tool calls at the end
    for tool_call in tool_calls:
        if tool_call:
            yield (UpdateType.TOOL_CALL, tool_call)


@retry(
    retry=retry_if_exception(
        lambda e: isinstance(e, (openai.RateLimitError, anthropic.RateLimitError))
    ),
    wait=wait_exponential(multiplier=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
@retry(
    retry=retry_if_exception(
        lambda e: isinstance(
            e,
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )
    ),
    wait=wait_exponential(multiplier=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
@observe()
async def async_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: Literal[tuple(MODELS)] = "gpt-4o-mini",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
) -> Tuple[str, List[ToolCall], bool]:
    """
    Non-streaming LLM call => returns (content, tool_calls, stop).
    """

    print("--------------------------------")
    print(f"Prompting {model} with {len(messages)} messages")
    print(dump_json([m.model_dump() for m in messages]))
    if tools: print("tools", tools.keys())
    print("--------------------------------")

    langfuse_context.update_current_observation(input=messages)

    if model.startswith("claude"):
        # Use the non-stream Anthropics helper
        return await async_anthropic_prompt(
            messages, system_message, model, response_model, tools
        )
    else:
        # Use existing OpenAI path
        return await async_openai_prompt(
            messages, system_message, model, response_model, tools
        )


@retry(
    retry=retry_if_exception(
        lambda e: isinstance(e, (openai.RateLimitError, anthropic.RateLimitError))
    ),
    wait=wait_exponential(multiplier=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
@retry(
    retry=retry_if_exception(
        lambda e: isinstance(
            e,
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )
    ),
    wait=wait_exponential(multiplier=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_prompt_stream(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: str,
    response_model: Optional[type[BaseModel]] = None,
    tools: Optional[Dict[str, Tool]] = None,
) -> AsyncGenerator[Tuple[UpdateType, str], None]:
    """
    Streaming LLM call => yields (UpdateType.ASSISTANT_TOKEN, partial_text).
    Add a similar function for OpenAI if you need streaming from GPT-based models.
    """

    async_prompt_stream_method = (
        async_anthropic_prompt_stream
        if model.startswith("claude")
        else async_openai_prompt_stream
    )

    async for chunk in async_prompt_stream_method(
        messages, system_message, model, response_model, tools
    ):
        yield chunk


def anthropic_prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_anthropic_prompt(messages, system_message, model, response_model, tools)
    )


def openai_prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_openai_prompt(messages, system_message, model, response_model, tools)
    )


def prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_prompt(messages, system_message, model, response_model, tools)
    )
