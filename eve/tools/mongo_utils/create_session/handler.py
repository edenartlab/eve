
from eve.auth import get_my_eden_user

# from ... import utils



from eve.agent2.handlers import async_playout_session

from eve.auth import get_my_eden_user
from eve.agent2.message import Channel
from eve.agent2.session_create import async_create_session


# from eve.agent2.dispatcher import async_run_dispatcher
from eve.tool import ToolContext



    
async def handler(context: ToolContext):
    user = get_my_eden_user()
    channel = Channel(type="discord", key="1003581679916548207")
    #prompt = "Eve is applying for a job to work at McDonalds, and GPTRumors is the interviewer."
    # prompt = "Eve and Mycos are competing to see who can generate the most outlandish image with Flux-Schnell. They keep one-upping each other. Make sure they alternate speaking. After Mycos speaks, Eve goes next, and vice-versa."
    prompt = context.args["prompt"]
    
    session = await async_create_session(user, channel, prompt)

    await async_playout_session(session, n_turns=context.args.get("n_turns", 10))
    
    return {
        "output": "https://edenartlab-stage-data.s3.amazonaws.com/d158dc1e5c62479489c1c3d119dd211bd56ba86a127359f7476990ec9e081cba.jpg"
    }
