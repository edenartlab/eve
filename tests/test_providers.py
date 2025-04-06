from eve.agent.thread import UserMessage
from eve.agent.llm import anthropic_prompt, openai_prompt, openrouter_prompt


def test_anthropic():
    messages = [
        UserMessage(content="Hello, can you help me?")
    ]
    content, tool_calls, stop = anthropic_prompt(messages=messages)
    print(content, tool_calls, stop)


def test_openai():
    messages = [
        UserMessage(content="Hello, can you help me?")
    ]
    content, tool_calls, stop = openai_prompt(messages=messages)
    print(content, tool_calls, stop)


def test_openrouter():
    messages = [
        UserMessage(content="Who created you? Who trained you?")
    ]
    content, tool_calls, stop = openrouter_prompt(
        messages=messages,
        model="google/gemini-2.0-flash-001"
    )
    print(content, tool_calls, stop)

