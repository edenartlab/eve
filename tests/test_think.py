from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent.run import prompt_thread
from eve.agent.think import think
from eve.agent.agent import Agent
from eve.tool import get_tools_from_mongo


def test_think():
    message = UserMessage(
        name="gene", 
        content="does someone know the year of the first moon landing?"
    )
    
    agent = Agent.load("eve")
    thread = Thread.from_mongo("6774249ff8d4aae98c89ac0f")

    result = think(
        agent=agent,
        thread=thread,
        user_message=message
    )
    
    print(result)

test_think()