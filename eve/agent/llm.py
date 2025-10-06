import os
import json
import asyncio
import logging
import anthropic
import openai
from enum import Enum
from typing import Optional, Dict, List, Union, Literal, Tuple, AsyncGenerator
from pydantic import BaseModel
from instructor.function_calls import openai_schema
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from ..tool import Tool
from .thread import UserMessage, AssistantMessage, ToolCall

logger = logging.getLogger(__name__)


MODELS = [
    "claude-sonnet-4-5",
    "claude-opus-4-1-20250805",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o1-mini",
    "o1",
    "o3",
    "google/gemini-2.0-flash-001"
]

DEFAULT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "claude-sonnet-4-5")

def get_anthropic_api_key():
    """Get the Anthropic API key from ANTHROPIC_API_KEY environment variable"""
    return os.getenv("ANTHROPIC_API_KEY")

class UpdateType(str, Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    END_PROMPT = "end_prompt"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_STOP = "assistant_stop"
    TOOL_CALL = "tool_call"


async def async_anthropic_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: Literal[tuple(MODELS)] = "claude-sonnet-4-5",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
):
    anthropic_client = anthropic.AsyncAnthropic(api_key=get_anthropic_api_key())

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

    # add tools / structure output
    if tools or response_model:
        tool_schemas = [
            t.anthropic_schema(exclude_hidden=True) for t in (tools or {}).values()
        ]
        if response_model:
            anthropic_schema = openai_schema(response_model).anthropic_schema
            tool_schemas.append(anthropic_schema)
            prompt["tool_choice"] = {"type": "tool", "name": response_model.__name__}

        # Add websearch tool:
        # websearch_tool = {
        #         "type": "web_search_20250305",
        #         "name": "web_search",
        #         "max_uses": 2
        # }
        # tool_schemas.append(websearch_tool)
        
        # cache all tools by checkpointing only the last tool
        tool_schemas[-1]["cache_control"] = {"type": "ephemeral"}

        prompt["tools"] = tool_schemas
        tool_names = [tool["name"] for tool in tool_schemas]

    import time

    start_time = time.time()

    # call Anthropic
    response = await anthropic_client.messages.create(**prompt)

    print("-----------RESPONSE USAGE---------------------")
    print(f"Time taken: {time.time() - start_time} seconds")
    print(response.usage)
    print("--------------------------------")

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


async def async_anthropic_prompt_stream(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: Literal[tuple(MODELS)] = "claude-sonnet-4-5",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
) -> AsyncGenerator[Tuple[UpdateType, str], None]:
    """Yields partial tokens (ASSISTANT_TOKEN, partial_text) for streaming."""
    anthropic_client = anthropic.AsyncAnthropic(api_key=get_anthropic_api_key())
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
            
        # Add websearch tool:
        # websearch_tool = {
        #         "type": "web_search_20250305",
        #         "name": "web_search",
        #         "max_uses": 2
        # }
        # tool_schemas.append(websearch_tool)
        
        # cache all tools - apply cache_control to each tool for full context caching
        for tool_schema in tool_schemas:
            tool_schema["cache_control"] = {"type": "ephemeral"}
            
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
    if system_message and model != "o1-mini":  # o1 does not support system messages
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
            model=model, messages=messages_json, tools=tools
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
async def async_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str],
    model: Literal[tuple(MODELS)] = DEFAULT_MODEL,
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
) -> Tuple[str, List[ToolCall], bool]:
    """
    Non-streaming LLM call => returns (content, tool_calls, stop).
    """
    if ("claude" in model) or ("anthropic" in model):
        print("Anthropic -> async_anthropic_prompt")
        return await async_anthropic_prompt(
            messages, system_message, model, response_model, tools
        )
    elif "gemini" in model:
        print("gemini -> async_openrouter_prompt")
        return await async_openrouter_prompt(
            messages, system_message, model, response_model, tools
        )
    else:
        # Use existing OpenAI path
        print("OpenAI -> async_openai_prompt")
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


def anthropic_prompt(
    messages,
    system_message="You are a helpful assistant.",
    model="claude-sonnet-4-5",
    response_model=None,
    tools=None,
):
    return asyncio.run(
        async_anthropic_prompt(messages, system_message, model, response_model, tools)
    )


def openai_prompt(
    messages,
    system_message="You are a helpful assistant.",
    model="gpt-4o-mini",
    response_model=None,
    tools=None,
):
    return asyncio.run(
        async_openai_prompt(messages, system_message, model, response_model, tools)
    )


def prompt(
    messages,
    system_message="You are a helpful assistant.",
    model="claude-sonnet-4-5",
    response_model=None,
    tools=None,
):
    return asyncio.run(
        async_prompt(messages, system_message, model, response_model, tools)
    )


async def async_openrouter_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: Literal[tuple(MODELS)] = "gpt-4o-mini",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
):
    print("Use openrouter", model)

    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY env is not set")

    messages_json = [item for msg in messages for item in msg.openai_schema()]
    # if system_message and model != "o1-mini":  # o1 does not support system messages
    #     messages_json = [{"role": "system", "content": system_message}] + messages_json

    # openai_client = openai.AsyncOpenAI()

    openai_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

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
            model=model, messages=messages_json, tools=tools
        )
        response = response.choices[0]
        content = response.message.content or ""
        tool_calls = [
            ToolCall.from_openai(t) for t in response.message.tool_calls or []
        ]
        stop = response.finish_reason == "stop"

        return content, tool_calls, stop


def openrouter_prompt(
    messages,
    system_message="You are a helpful assistant.",
    model="gpt-4o-mini",
    response_model=None,
    tools=None,
):
    return asyncio.run(
        async_openrouter_prompt(messages, system_message, model, response_model, tools)
    )
