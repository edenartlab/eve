from eve.agent.thread import UserMessage
from eve.agent.llm import prompt


def test_llm():    
    
    messages = [
        UserMessage(content="Hello, can you help me?")
    ]

    results = prompt(
        messages=messages,
        system_message="You are a helpful assistant.",
        model="gpt-4o-mini"
    )

    print(results)