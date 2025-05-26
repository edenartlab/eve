import litellm
import pytest
from unittest.mock import AsyncMock, patch
from bson import ObjectId
from typing import Dict
from pydantic import BaseModel
from litellm import ModelResponse

from eve.agent.session.models import LLMContextMetadata, LLMTraceMetadata
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
    ChatMessage(role="user", content="Hello", session=MOCK_SESSION_ID),
    ChatMessage(role="assistant", content="Hi there!", session=MOCK_SESSION_ID),
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


MOCK_TOOLS = [
    MockTool(
        key="test_tool",
        name="test_tool",
        description="A test tool",
        output_type="string",
        cost_estimate="0.0001",
        model=MockToolModel,
        handler="local",
        parameters={"input": {"type": "string", "description": "Input text"}},
    )
]


# Mocked tests
@pytest.mark.asyncio
async def test_validate_input_valid_model():
    context = LLMContext(messages=MOCK_MESSAGES)
    config = LLMConfig(model="gpt-4o-mini")
    validate_input(context, config)  # Should not raise


@pytest.mark.asyncio
async def test_validate_input_invalid_model():
    context = LLMContext(messages=MOCK_MESSAGES)
    config = LLMConfig(model="invalid-model")
    with pytest.raises(ValueError, match="Model invalid-model is not supported"):
        validate_input(context, config)


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
        session_id=MOCK_SESSION_ID,
        initiating_user_id=MOCK_USER_ID,
    )
    result = construct_observability_metadata(context)
    assert result["session_id"] == MOCK_SESSION_ID
    assert result["initiating_user_id"] == MOCK_USER_ID


@pytest.mark.asyncio
async def test_async_prompt_mocked():
    mock_response = "Mocked response"
    mock_completion = AsyncMock(return_value=mock_response)

    with patch("eve.agent.session.session_llm.completion", mock_completion):
        context = LLMContext(messages=MOCK_MESSAGES)
        config = LLMConfig()
        result = await async_prompt(context, config)

        assert result == mock_response
        mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_async_prompt_stream_mocked():
    mock_responses = ["Part 1", "Part 2", "Part 3"]

    async def mock_stream():
        for response in mock_responses:
            yield response

    mock_completion = AsyncMock(return_value=mock_stream())

    with patch("eve.agent.session.session_llm.completion", mock_completion):
        context = LLMContext(messages=MOCK_MESSAGES)
        config = LLMConfig()
        stream = async_prompt_stream(context, config)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == mock_responses


@pytest.mark.live
@pytest.mark.asyncio
async def test_async_prompt():
    metadata = LLMContextMetadata(
        trace_name="test_async_prompt",
        trace_metadata=LLMTraceMetadata(
            session_id=str(MOCK_SESSION_ID),
            initiating_user_id=str(MOCK_USER_ID),
        ),
    )
    context = LLMContext(
        messages=MOCK_MESSAGES,
        metadata=metadata,
    )
    result = await async_prompt(context)
    assert isinstance(result, ModelResponse)


@pytest.mark.live
@pytest.mark.asyncio
async def test_async_prompt_stream_real():
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


# @pytest.mark.live
# @pytest.mark.asyncio
# async def test_async_prompt_with_tools_real():
#     context = LLMContext(messages=MOCK_MESSAGES, tools=MOCK_TOOLS)
#     config = LLMConfig()
#     result = await async_prompt(context, config)
#     assert isinstance(result, str)
#     assert len(result) > 0
