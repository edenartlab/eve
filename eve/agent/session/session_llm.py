from dataclasses import dataclass
from bson import ObjectId
from litellm import completion
import litellm
from typing import List, AsyncGenerator, Optional

from eve.agent.session.session import ChatMessage
from eve.tool import Tool

litellm.success_callback = ["langfuse"]

supported_models = ["gpt-4o-mini", "gpt-4o"]


@dataclass
class LLMContext:
    session_id: ObjectId
    messages: List[ChatMessage]
    tools: List[Tool]
    initiating_user_id: Optional[ObjectId] = None


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"


def construct_observability_metadata(context: LLMContext):
    return {
        "session_id": context.session_id,
        "initiating_user_id": context.initiating_user_id,
    }


def construct_messages(context: LLMContext):
    return [msg.openai_schema() for msg in context.messages]


def construct_tools(context: LLMContext):
    if not context.tools:
        return None
    return [tool.openai_schema(exclude_hidden=True) for tool in context.tools]


async def async_prompt(
    context: LLMContext,
    config: Optional[LLMConfig] = None,
) -> str:
    if not config:
        config = LLMConfig()
    response = await completion(
        model=config.model,
        messages=construct_messages(context),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
    )
    return response


async def async_prompt_stream(
    context: LLMContext,
    config: Optional[LLMConfig] = None,
) -> AsyncGenerator[str, None]:
    if not config:
        config = LLMConfig()
    response = await completion(
        model=config.model,
        messages=construct_messages(context),
        metadata=construct_observability_metadata(context),
        tools=construct_tools(context),
        stream=True,
    )
    for part in response:
        yield part
