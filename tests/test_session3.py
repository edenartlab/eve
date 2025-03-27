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





import os
from eve.user import User
from eve.agent2 import Agent
from eve.agent2.message import ChatMessage
# from eve.tools import Tool
from eve.eden_utils import load_template

system_template = load_template("system2")
knowledge_reply_template = load_template("knowledge_reply")
models_instructions_template = load_template("models_instructions")
model_template = load_template("model_doc")



"""

old flow - receive message, prompt agent, stop

sessions
 - create session
 - advance session (run dispatcher, schedule next speaker, prompt agent)
 - playout whole session (N x advance session)



"""


from eve.agent2.handlers import async_playout_session










async def test_session():
    user = get_my_eden_user()
    channel = Channel(type="discord", key="1268682080263606443")
    #prompt = "Eve is applying for a job to work at McDonalds, and GPTRumors is the interviewer."
    prompt = "Eve and Mycos are competing to see who can generate the most outlandish image with Flux-Schnell. They keep one-upping each other. Make sure they alternate speaking. After Mycos speaks, Eve goes next, and vice-versa."

    # prompt = "gene2 and xander2 are debating what they should do with eden"
    prompt = "hai-dai, xander2, jordan, and chebel-01 are collaborating on a visual story that represents a dream hai dai had. they use flux-schnell for a while, using *themselves* (through their model/lora) as the subject, and when they're happy, they all agree on the best one and they animate the best image they made with runway."

    prompt = "hai-dai, xander2, jordan, and chebel-01 are collaborating on a visual story that represents a dream hai dai had. they use flux-schnell for a while, using *themselves* (through their model/lora) as the subject, and when they're happy, they all agree on the best one and they animate the best image they made with runway."
    
    prompt = "chebel-01 and gene2 are playing a game. Each one takes turns drawing an image of themselves, using their model/lora on flux_dev_lora. They then attempt to mimic the other's style, like a game of telephone."

    prompt = "xander, hai-dai, and jaimill are trying to one up each other in terms of the most outrageous image they can make with flux_dev_lora. each one uses their own model/lora"

    prompt = "chebel-01 is drawing various pictures of herself using her dault lora model"

    prompt = "vincent and joey-v1 are having a discussion about something very dark and sinister, but kind of laughing about it, and illustrating it with images."

    prompt = "Ygor2 and maggi1 are having a magical tea ceremony in the desert and illustrating it with images. And after they're done making 5 images, they animate them all with runway, and then composite them into one final film together."

    prompt = "xander2, chebel-01, gene2, jordan, Ygor2, and maggi1 are debating the future of Mars College after the symposium meeting. The debate is very heated and contentious, and they are finding it very hard to agree on anything. They are all very disagreeable and disagree about everything and start yelling at each other!!!"

    prompt = "maggi1 is trying to make an art installation that is capable of speaking with a real voice but is not sure how to do that. she asks xander2, Ygor2, and joey-v1 for help."


    session = await async_create_session(user, channel, prompt)

    print("--- session created ---")

    await async_playout_session(session, n_turns=10)





async def test_session2():
    user = get_my_eden_user()

    # Create a new session
    request = MessageRequest(
        user_id=str(user.id),
        session_id=None, #"67d115430deaf0504325447a",
        message=ChatMessage(
            content="Eve is applying for a job to work at McDonalds, and GPTRumors is the interviewer."
        ),
        update_config=None
    )

    result = await async_receive_message(request)    
    print(result)




from eve.user import User
from eve.agent2 import Agent
from eve.agent2.message import ChatMessage
# from eve.tools import Tool
async_playout_session
from eve.api.helpers import emit_update


from eve.agent2.llm import async_prompt




if __name__ == "__main__":
    import asyncio
    asyncio.run(test_session())

