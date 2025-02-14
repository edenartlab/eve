from eve.llm import prompt_thread, UserMessage, AssistantMessage, async_think
from eve.tool import get_tools_from_mongo
from eve.auth import get_my_eden_user

from eve.agent import Agent
from eve.thread import Thread

# todo: since prompt_thread handles exceptions, this won't actually fail if there are errors
def test_prompting():
    user = get_my_eden_user()

    agent = Agent.load("eve")
    tools = agent.get_tools()
    thread = agent.request_thread()

    messages = [
        UserMessage(
            content="make another picture of a fancy dog"
        )
    ]

    for msg in prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages, 
        tools=tools,
        model="gpt-4o-mini"
    ):
        print(msg)


def test_prompting2():
    user = get_my_eden_user()

    messages = [
        UserMessage(name="jim", content="i have an apple."),
        UserMessage(name="kate", content="the capital of france is paris?"),
        UserMessage(name="morgan", content="what is even going on here? im so confused."),
        UserMessage(name="kate", content="what is my name?"),
    ]

    agent = Agent.load("eve")
    tools = agent.get_tools()
    thread = agent.request_thread()

    for msg in prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        model="gpt-4o-mini"
    ):
        print(msg)


def test_valid_messages():
    user = get_my_eden_user()

    messages = [
        UserMessage(name="alice", content="i am alice"),
        UserMessage(name="bob", content="i am bob."),
        AssistantMessage(content="now i will speak."),
        AssistantMessage(content="i am eve. who am i? let me say that again, but with exclamation marks at the end. and i'm going to rename myself Esmerelda."),
        # UserMessage(name="kate", content="Eve what is my name?"),
    ]

    agent = Agent.load("eve")
    tools = agent.get_tools()
    thread = agent.request_thread()

    for msg in async_think(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        force_reply=False,
        model="gpt-4o-mini"
    ):
        print(msg)




def test_think():
    user = get_my_eden_user()

    messages = [
        UserMessage(name="gene", content="eve, make a picture of a golden retriever playing chess against Vitalik Buterin in impressionist style."),
        # UserMessage(name="gene", content="eve, tell me a funny joke"),
        # UserMessage(name="gene", content="alright nice."),
    ]

    agent = Agent.load("eve")
    tools = agent.get_tools()
    # thread = agent.request_thread()
    thread = Thread.from_mongo("6774249ff8d4aae98c89ac0f")

    for msg in prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        # force_reply=True,
        # model="gpt-4o-mini"
        model="claude-3-5-sonnet-20240620"
    ):
        print(msg)


test_think()