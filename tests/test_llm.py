from eve.llm import prompt_thread, UserMessage, AssistantMessage
from eve.tool import get_tools_from_mongo
from eve.app.auth.auth import get_my_eden_user

from eve.agent import Agent
from eve.thread import Thread

# todo: since prompt_thread handles exceptions, this won't actually fail if there are errors
def test_prompting():
    user = get_my_eden_user(db="STAGE")

    agent = Agent.load("eve", db="STAGE")
    tools = agent.get_tools()
    thread = agent.request_thread()

    messages = [
        UserMessage(
            content="make another picture of a fancy dog"
        )
    ]

    for msg in prompt_thread(
        db="STAGE",
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages, 
        tools=tools,
        model="gpt-4o-mini"
    ):
        print(msg)


def test_prompting2():
    user = get_my_eden_user(db="STAGE")

    messages = [
        UserMessage(name="jim", content="i have an apple."),
        UserMessage(name="kate", content="the capital of france is paris?"),
        UserMessage(name="morgan", content="what is even going on here? im so confused."),
        UserMessage(name="kate", content="what is my name?"),
    ]

    agent = Agent.load("eve", db="STAGE")
    tools = agent.get_tools()
    thread = agent.request_thread()

    for msg in prompt_thread(
        db="STAGE",
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages,
        tools=tools,
        model="gpt-4o-mini"
    ):
        print(msg)
