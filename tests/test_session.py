from eve.auth import get_my_eden_user
from eve.agent3.message import UserMessage, AssistantMessage
from eve.agent3.handlers import async_receive_message

# from eve.agent3.run_thread import prompt_thread, async_prompt_thread
# from eve.agent3.think import async_think
# from eve.agent3 import Agent

# from eve.agent import Tool

async def test_session():

    user = get_my_eden_user()
    message = UserMessage(name="kate", content="make a picture of a cat")

    print(user)
    print(message)

    await async_receive_message(user.id, "123", message)
    


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_session())