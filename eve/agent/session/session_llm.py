import logging
import os

from eve.agent.thread import ChatMessage

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

import json
from litellm import completion
import litellm
from typing import Callable, List, AsyncGenerator, Optional

from eve.agent.session.models import (
    LLMContext,
    LLMConfig,
    LLMContextMetadata,
    LLMResponse,
    LLMTraceMetadata,
    ToolCall,
)


if os.getenv("LANGFUSE_TRACING_ENVIRONMENT"):
    litellm.success_callback = ["langfuse"]

supported_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-haiku-latest",
    "gemini-2.0-flash",
    "gemini/gemini-2.5-flash-preview-04-17",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
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
        "trace_name": context.metadata.trace_name,
        "generation_name": context.metadata.generation_name,
    }
    if context.metadata.trace_metadata:
        metadata["trace_metadata"] = context.metadata.trace_metadata.model_dump()
    return metadata


def construct_tools(context: LLMContext) -> Optional[List[dict]]:
    if not context.tools:
        return None
    return [tool.openai_schema(exclude_hidden=True) for tool in context.tools.values()]


async def async_run_tool_call(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
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


def prepare_messages(messages: List[ChatMessage]) -> List[dict]:
    messages = [schema for msg in messages for schema in msg.openai_schema()]
    return messages


async def async_prompt_litellm(
    context: LLMContext,
) -> LLMResponse:
    print("=> model", context.config.model)
    response = completion(
        model=context.config.model,
        messages=prepare_messages(context.messages),
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
        tokens_spent=response.usage.total_tokens,
    )


async def async_prompt_stream_litellm(
    context: LLMContext,
) -> AsyncGenerator[str, None]:
    response = await litellm.acompletion(
        model=context.config.model,
        messages=prepare_messages(context.messages),
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
