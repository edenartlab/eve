from bson import ObjectId
from eve.agent.session import Session, SessionMessage


from eve.auth import get_my_eden_user
from eve.agent.thread import UserMessage, AssistantMessage, Thread
from eve.agent2.handlers import async_run_session


from eve.agent.think import async_think
from eve.agent import Agent

from eve.agent import Tool


from eve.api.handlers import handle_session_message
from eve.api.handlers import SessionMessageRequest


from eve.agent2.handlers import MessageRequest
from eve.agent2.message import ChatMessage




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
    
    result = await async_run_session(request)
    
    print(result)



if __name__ == "__main__":
    import asyncio
    asyncio.run(test_session2())