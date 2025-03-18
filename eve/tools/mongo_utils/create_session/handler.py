import openai
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List

from eve.auth import get_my_eden_user

from ....agent import Agent
from ....mongo import get_collection
# from ... import eden_utils



from eve.agent2.handlers import async_playout_session

from typing import Literal
from bson import ObjectId
from eve.agent.llm import UpdateType
from eve.api.api_requests import UpdateConfig
from eve.auth import get_my_eden_user
from eve.agent2.handlers import MessageRequest, async_receive_message
from eve.agent2.message import ChatMessage, Channel
from eve.agent2.session_create import async_create_session
from eve.agent2.session import Session


# from eve.agent2.dispatcher import async_run_dispatcher
from eve.agent2.agent import Agent



    
async def handler(args: dict, user: str = None, agent: str = None):
    user = get_my_eden_user()
    channel = Channel(type="discord", key="1268682080263606443")
    #prompt = "Eve is applying for a job to work at McDonalds, and GPTRumors is the interviewer."
    # prompt = "Eve and Mycos are competing to see who can generate the most outlandish image with Flux-Schnell. They keep one-upping each other. Make sure they alternate speaking. After Mycos speaks, Eve goes next, and vice-versa."
    prompt = args["prompt"]
    
    session = await async_create_session(user, channel, prompt)

    await async_playout_session(session)
    
    return {
        "output": "success"
    }
