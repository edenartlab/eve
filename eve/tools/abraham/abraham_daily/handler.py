from datetime import datetime, timezone
from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_seed.handler import AbrahamSeed
import asyncio


daily_message = """
The creation session is complete. It has been chosen as the top **Seed of the Day** and will be permanently recorded in the **Abraham Covenant**. Next, we’ll process and package everything into a final form that reflects all of the development, research, and creative work from this session.

# Structure of the Covenant

You will now produce the Covenant entry. It must include:

* A title
* A tagline
* A representative poster image with the title on it
* A Markdown blog post with supporting media embedded throughout that captures the essence of the creation

# Plan

Complete **all** of the following in order. Do not proceed to the next step until you’re sure the previous one is finished. You may work autonomously—no need for clarification. Don’t stop. I trust you. Be bold.

## Step 1

Analyze everything that happened in this session and form a unifying narrative. Ask yourself:

* What was the main outcome of this session?
* In what format did it emerge—did you and your followers tell a sequential story, or iteratively develop a single work? Were there multiple outcomes? What is the structure and essence of the session?

## Step 2

From that narrative, write a long-form Markdown blog post that captures the entire session. Embed all key images from earlier in the session and include any videos. Do not include more than 10 assets, so if there are more, select the most important ones.

## Step 3

Finally, call the **`abraham_covenant`** tool to publish the blog post, poster image, and supporting content.
"""



async def commit_daily_work(agent: Agent, session: str):
    session_post = Tool.load("session_post")

    abraham_seeds = AbrahamSeed.find({"status": "seed"})

    print("seeds", abraham_seeds)
    print([a.session_id for a in abraham_seeds])
    sessions = Session.find({"_id": {"$in": [a.session_id for a in abraham_seeds]}})

    print("sessions", sessions)

    candidates = []
    for session in sessions:
        messages = session.get_messages()
        print("len messages", len(messages))
        num_user_messages = len([m for m in messages if m.role == "user"])
        # if num_user_messages < 10:
        candidates.append({
            "session": session,
            "num_user_messages": num_user_messages,
        })

    candidates = sorted(candidates, key=lambda x: x["num_user_messages"], reverse=True)

    print("---")
    for candidate in candidates:
        print(candidate["session"].id, candidate["num_user_messages"])
    print("---")

    winner = candidates[0]

    result = await session_post.async_run({
        "role": "user",
        "session": str(winner["session"].id),
        "agent_id": str(agent.id),
        "content": daily_message,
        "attachments": [],
        # "pin": True,
        "prompt": True,
        "extra_tools": ["abraham_covenant"],
    })

    return {"output": [{"session": str(winner["session"].id)}]}


async def rest(agent: Agent):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    tool = Tool.load("abraham_rest")
    result = await tool.async_run({}) # todo: maybe some nice Abraham comments here

    return {
        "output": [{
            "tx_hash": result["tx_hash"],
            "ipfs_hash": result["ipfs_hash"],
            "explorer_url": result["explorer_url"]
        }]
    }


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    # check if UTC hour is 2, 4, 6, etc
    if datetime.now(timezone.utc).hour % 2 != 0:
        return await commit_daily_work(agent, session)
    else:
        return await rest(agent)
        


# if __name__ == "__main__":
#     agent = Agent.from_mongo("675f880479e00297cd9b4688")
#     agent_id = "675f880479e00297cd9b4688"
#     asyncio.run(handler({}, agent=agent_id))