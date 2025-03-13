from bson import ObjectId
from eve.agent.session import Session, SessionMessage


from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent2.handlers import async_receive_message


from eve.agent.think import async_think
from eve.agent import Agent

from eve.agent import Tool


from eve.api.handlers import handle_session_message
from eve.api.handlers import SessionMessageRequest


from eve.agent2.handlers import MessageRequest
from eve.agent2.message import ChatMessage, Channel




async def test_session2():
    user = get_my_eden_user()

    # Create a new session
    request = MessageRequest(
        user_id=str(user.id),
        session_id=None, #"67d115430deaf0504325447a",
        message=ChatMessage(
            content="Eve is applying for a job to work at McDonalds, and Banny is the interviewer."
        ),
        update_config=None
    )
    
    result = await async_receive_message(request)
    
    print(result)



from eve.agent2.session_create import async_create_session




async def test_create_session():
    user = get_my_eden_user()

    channel = Channel(type="discord", key="1268682080263606443")

    prompt = "Eve is applying for a job to work at McDonalds, and Banny is the interviewer."

    result = await async_create_session(user, channel, prompt)
    
    print(result)



if __name__ == "__main__":
    import asyncio
    asyncio.run(test_create_session())