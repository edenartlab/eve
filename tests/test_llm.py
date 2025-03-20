"""

test_base
test_base_tool
test_tool
test_llm
 - no threads
test_thread
 - thread + llm
test_api

"""


from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent.run_thread import prompt_thread
from eve.agent.think import think
from eve.agent.llm import UpdateType
from eve.agent import Agent


def test_llm():
    user = get_my_eden_user()

    agent = Agent.load("eve")
    tools = agent.get_tools()
    thread = agent.request_thread(key=f"test")

    messages = [
        UserMessage(
            content="eve, make another picture of a fancy dog"
        )
    ]

    for update in prompt_thread(
        user=user,
        agent=agent,
        thread=thread,
        user_messages=messages, 
        tools=tools,
        model="gpt-4o-mini"
    ):
        print(update)
        assert update.type != UpdateType.ERROR


def test_think():
    agent = Agent.load("eve")
    thread = agent.request_thread(key=f"test")
    
    thought = think(
        agent=agent,
        thread=thread,
        user_message=UserMessage(content="eve, i want to talk to you"),
        model="gpt-4o-mini"
    )
    
    print(thought)
    assert thought.intention == "reply"


