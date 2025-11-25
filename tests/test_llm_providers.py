from datetime import datetime

import pytest
import pytz

from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.system_template import system_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext


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
        config=LLMConfig(model="gemini-2.5-flash"),
    )

    response = await async_prompt(context)
    print(f"Google Gemini Response: {response.content}")
    assert response.content
    assert len(response.content) > 0
