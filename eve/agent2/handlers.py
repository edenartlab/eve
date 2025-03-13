from pydantic import BaseModel
from typing import Optional

from ..api.api_requests import UpdateConfig
from .dispatcher import async_run_dispatcher
from .message import ChatMessage
from .session import Session
from .agent import Agent


# todo: force-reply, use-thinking, model, user-is-bot
class MessageRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: ChatMessage
    update_config: Optional[UpdateConfig] = None    
    # force_reply: bool = False
    # use_thinking: bool = True
    # model: Optional[str] = None
    # user_is_bot: Optional[bool] = False




# @sentry_transaction(op="llm.prompt", name="async_prompt_thread")
async def async_receive_message(
    request: MessageRequest
):
    # if USE_RATE_LIMITS:
    #     await RateLimiter.check_chat_rate_limit(user.id, None)


    # get or create session
    if request.session_id is None:
        
        
        
        
        chat = session.get_chat_log(25)

        latest_user_message = request.message.content

        prompt = dispatcher_template.render(
            agents=agents_text,
            scenario="session.scenario",
            current="session.current",
            chat_log=chat,
            latest_message=latest_user_message,
        )

        
        
        
        
        
        
        
        session = Session(name="New Session 3")
        # figure out initial agents
        session.agents = [
            Agent.load("eve").id,
            Agent.load("banny").id,
        ]
        session.save()
    
    
    else:
        session = Session.from_mongo(request.session_id)
        







    result = await async_run_dispatcher(session, request.message)
    print(result)


    speakers = result.speakers or []
    for speaker in speakers:
        agent = Agent.load(speaker)
        print("get agent", agent)
        # tools = agent.get_tools()
        # thread = agent.request_thread()
        # async for msg in async_prompt_thread(
        #     user, 
        #     agent, 
        #     thread, 
        #     user_messages, 
        #     tools, 
        #     force_reply=True, 
        #     use_thinking=False, 
        #     model=DEFAULT_MODEL
        # ):
        #     print(msg)

    return result
