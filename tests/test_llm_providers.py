from datetime import datetime

import pytest
import pytz

from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.system_template import system_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext


class MockTool:
    """Simple mock tool for testing LLM provider tool calling."""

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def openai_schema(self, exclude_hidden: bool = False) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def anthropic_schema(self, exclude_hidden: bool = False) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def gemini_schema(self, exclude_hidden: bool = False) -> dict:
        """Gemini function declaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# Create a mock weather tool for testing
def create_weather_tool():
    return MockTool(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. 'Paris, France'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_openai_gpt():
    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant.",
        tools=None,
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="Tell me which company trained you, and describe what you see in this image",
                attachments=[
                    "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg"
                ],
            ),
        ],
        config=LLMConfig(model="gpt-4o-mini"),
    )

    response = await async_prompt(context)
    print(f"OpenAI GPT Response: {response.content}")
    assert response.content
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.live
async def test_anthropic_claude():
    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant.",
        tools=None,
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="Tell me which company trained you, and describe what you see in this image",
                attachments=[
                    "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg"
                ],
            ),
        ],
        config=LLMConfig(model="claude-haiku-4-5"),
    )

    response = await async_prompt(context)
    print(f"Anthropic Claude Response: {response.content}")
    assert response.content
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.live
async def test_google_gemini():
    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant.",
        tools=None,
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="Tell me which company trained you, and describe what you see in this image",
                attachments=[
                    "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg"
                ],
            ),
        ],
        config=LLMConfig(model="gemini-3-flash-preview"),
    )

    response = await async_prompt(context)
    print(f"Google Gemini Response: {response.content}")
    assert response.content
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.provider_gemini
async def test_google_gemini_tool_calling():
    """Test Gemini tool calling functionality."""
    weather_tool = create_weather_tool()

    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant that can check the weather.",
        tools={"get_weather": weather_tool},
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="What's the weather like in Paris?",
            ),
        ],
        config=LLMConfig(model="gemini-3-flash-preview"),
        tools={"get_weather": weather_tool},
    )

    response = await async_prompt(context)
    print(f"Google Gemini Tool Calling Response: {response}")
    print(f"Tool calls: {response.tool_calls}")

    # The model should either respond with text or request a tool call
    assert response.content or response.tool_calls
    if response.tool_calls:
        assert len(response.tool_calls) > 0
        tool_call = response.tool_calls[0]
        assert tool_call.tool == "get_weather"
        assert "location" in tool_call.args


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.provider_gemini
async def test_google_gemini_thinking():
    """Test Gemini thinking/reasoning functionality."""
    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant that thinks carefully about problems.",
        tools=None,
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="What is 17 * 23? Think step by step.",
            ),
        ],
        config=LLMConfig(
            model="gemini-3-flash-preview",
            reasoning_effort="medium",  # This maps to thinking_level
        ),
    )

    response = await async_prompt(context)
    print(f"Google Gemini Thinking Response: {response.content}")
    print(f"Thought: {response.thought}")

    assert response.content
    assert "391" in response.content  # 17 * 23 = 391


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.provider_anthropic
async def test_anthropic_tool_calling():
    """Test Anthropic tool calling functionality."""
    weather_tool = create_weather_tool()

    system_message = system_template.render(
        name="Test Assistant",
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona="You are a helpful AI assistant that can check the weather.",
        tools={"get_weather": weather_tool},
    )

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(
                role="user",
                content="What's the weather like in Tokyo?",
            ),
        ],
        config=LLMConfig(model="claude-haiku-4-5"),
        tools={"get_weather": weather_tool},
    )

    response = await async_prompt(context)
    print(f"Anthropic Tool Calling Response: {response}")
    print(f"Tool calls: {response.tool_calls}")

    # The model should either respond with text or request a tool call
    assert response.content or response.tool_calls
    if response.tool_calls:
        assert len(response.tool_calls) > 0
        tool_call = response.tool_calls[0]
        assert tool_call.tool == "get_weather"
        assert "location" in tool_call.args
