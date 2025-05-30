import litellm
import pytest
from bson import ObjectId
from typing import Dict
from pydantic import BaseModel
from litellm import ModelResponse

from eve.agent.session.models import LLMContextMetadata, LLMResponse, LLMTraceMetadata
from eve.agent.session.session_llm import (
    LLMContext,
    LLMConfig,
    async_prompt,
    async_prompt_stream,
    validate_input,
    construct_messages,
    construct_tools,
    construct_observability_metadata,
)
from eve.agent.session.session import ChatMessage
from eve.tool import Tool
from eve.task import Task

# Test data
MOCK_SESSION_ID = ObjectId()
MOCK_USER_ID = ObjectId()
MOCK_MESSAGES = [
    ChatMessage(
        role="user", sender=MOCK_USER_ID, content="Hello", session=MOCK_SESSION_ID
    ),
    ChatMessage(
        role="assistant",
        sender=MOCK_USER_ID,
        content="Hi there!",
        session=MOCK_SESSION_ID,
    ),
]


class MockToolModel(BaseModel):
    input: str


class MockTool(Tool):
    async def async_run(self, args: Dict, mock: bool = False):
        return {"output": "mock output"}

    async def async_start_task(
        self,
        user_id: str,
        agent_id: str,
        args: Dict,
        mock: bool = False,
        public: bool = False,
        is_client_platform: bool = False,
    ):
        return "mock_handler_id"

    async def async_wait(self, task: Task):
        return {"output": "mock output"}

    async def async_cancel(self, task: Task, force: bool = False):
        pass


MOCK_TOOLS = {
    "MockTool": MockTool(
        key="test_tool",
        name="test_tool",
        description="A test tool",
        output_type="string",
        cost_estimate="0.0001",
        model=MockToolModel,
        handler="local",
        parameters={"input": {"type": "string", "description": "Input text"}},
    )
}


# Mocked tests
@pytest.mark.asyncio
async def test_validate_input_valid_model():
    config = LLMConfig(model="gpt-4o-mini")
    context = LLMContext(messages=MOCK_MESSAGES, config=config)
    validate_input(context)  # Should not raise


@pytest.mark.asyncio
async def test_validate_input_invalid_model():
    config = LLMConfig(model="invalid-model")
    context = LLMContext(messages=MOCK_MESSAGES, config=config)
    with pytest.raises(ValueError, match="Model invalid-model is not supported"):
        validate_input(context)


def test_construct_messages():
    context = LLMContext(messages=MOCK_MESSAGES)
    result = construct_messages(context)
    assert len(result) == 2
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Hi there!"


def test_construct_tools():
    context = LLMContext(messages=MOCK_MESSAGES, tools=MOCK_TOOLS)
    result = construct_tools(context)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "MockToolModel"


def test_construct_tools_none():
    context = LLMContext(messages=MOCK_MESSAGES)
    result = construct_tools(context)
    assert result is None


def test_construct_observability_metadata():
    context = LLMContext(
        messages=MOCK_MESSAGES,
        metadata=LLMContextMetadata(
            session_id="test_session",
            generation_name="test_generation",
            trace_name="test_trace",
        ),
    )
    result = construct_observability_metadata(context)
    assert result["session_id"] == "test_session"
    assert result["trace_name"] == "test_trace"
    assert result["generation_name"] == "test_generation"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,provider_mark",
    [
        pytest.param("gpt-4o-mini", None, id="openai"),
        pytest.param(
            "claude-3-5-haiku-latest",
            "provider_anthropic",
            marks=pytest.mark.provider_anthropic,
            id="anthropic",
        ),
        pytest.param(
            "gemini/gemini-2.5-flash-preview-04-17",
            "provider_gemini",
            marks=pytest.mark.provider_gemini,
            id="gemini",
        ),
    ],
)
async def test_async_prompt(model, provider_mark, request):
    metadata = LLMContextMetadata(
        trace_name=f"test_async_prompt_{model.replace('-', '_')}",
        trace_metadata=LLMTraceMetadata(
            session_id=str(MOCK_SESSION_ID),
            initiating_user_id=str(MOCK_USER_ID),
        ),
    )
    context = LLMContext(
        messages=MOCK_MESSAGES,
        metadata=metadata,
        config=LLMConfig(model=model),
    )
    result = await async_prompt(context)
    assert isinstance(result, LLMResponse)


@pytest.mark.live
@pytest.mark.asyncio
async def test_async_prompt_stream():
    metadata = LLMContextMetadata(
        trace_name="test_async_prompt_stream",
        trace_metadata=LLMTraceMetadata(
            session_id=str(MOCK_SESSION_ID),
            initiating_user_id=str(MOCK_USER_ID),
        ),
    )
    context = LLMContext(
        messages=MOCK_MESSAGES,
        metadata=metadata,
    )
    stream = async_prompt_stream(context)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    response = litellm.stream_chunk_builder(chunks, messages=MOCK_MESSAGES)
    assert response is not None
    assert response.choices[0].message.content is not None
    assert isinstance(response, ModelResponse)
