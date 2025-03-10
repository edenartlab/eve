from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent.run_thread import prompt_thread, async_prompt_thread
from eve.agent.think import async_think
from eve.agent.agent import Agent

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


from eve.agent.agent import Tool

async def test_sub():

    user = get_my_eden_user()

    messages = [
        UserMessage(name="kate", content="make a picture of a cat"),
    ]

    agent = Agent.load("eve")
    agent.tools = {
        "txt2img": {},
    }
    tools = agent.get_tools()
    print("TOOLS", tools)
    print("TOOLS", tools.keys())
    thread = agent.request_thread()

    async for msg in async_prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        force_reply=True,
        use_thinking=False,
        model="gpt-4o-mini"
    ):
        print(msg)


from eve.agent.run_thread import async_run_session
from eve.agent.session import Session

async def test_dispatch_run():

    user = get_my_eden_user()
    
    session = Session.load("test-session")
    
    messages = [
        UserMessage(name="kate", content="make a picture of a cat"),
    ]

    result = await async_run_session(
        user=user,
        session=session,
        user_messages=messages,
    )
    print(result)


import asyncio
asyncio.run(test_dispatch_run())
