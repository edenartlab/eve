from dataclasses import dataclass
from bson import ObjectId
from litellm import completion
import litellm
from typing import Callable, List, AsyncGenerator, Optional

from eve.agent.session.session import ChatMessage
from eve.tool import Tool

litellm.success_callback = ["langfuse"]

supported_models = ["gpt-4o-mini", "gpt-4o"]


@dataclass
class LLMContext:
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    session_id: Optional[ObjectId] = None
    initiating_user_id: Optional[ObjectId] = None


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"


def construct_observability_metadata(context: LLMContext):
    return {
        "session_id": context.session_id,
        "initiating_user_id": context.initiating_user_id,
    }


def construct_messages(context: LLMContext) -> List[dict]:
    return [msg.openai_schema() for msg in context.messages]


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    if not context.tools:
        return None
    return [tool.openai_schema(exclude_hidden=True) for tool in context.tools]


async def async_prompt_litellm(
    context: LLMContext,
    config: LLMConfig,
) -> str:
    response = await completion(
        model=config.model,
        messages=construct_messages(context),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
    )
    return response


async def async_prompt_stream_litellm(
    context: LLMContext,
    config: LLMConfig,
) -> AsyncGenerator[str, None]:
    response = await completion(
        model=config.model,
        messages=construct_messages(context),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
        stream=True,
    )
    for part in response:
        yield part


DEFAULT_LLM_HANDLER = async_prompt_litellm
DEFAULT_LLM_STREAM_HANDLER = async_prompt_stream_litellm


async def async_prompt(
    context: LLMContext,
    config: Optional[LLMConfig] = LLMConfig(),
    handler: Optional[Callable[[LLMContext, LLMConfig], str]] = DEFAULT_LLM_HANDLER,
) -> str:
    return await handler(context, config)


async def async_prompt_stream(
    context: LLMContext,
    config: Optional[LLMConfig] = LLMConfig(),
    handler: Optional[
        Callable[[LLMContext, LLMConfig], AsyncGenerator[str, None]]
    ] = DEFAULT_LLM_STREAM_HANDLER,
) -> AsyncGenerator[str, None]:
    return await handler(context, config)
