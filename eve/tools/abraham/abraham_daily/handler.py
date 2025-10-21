from datetime import datetime, timezone
from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal
import math

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_seed.handler import AbrahamSeed
import asyncio


def user_score(msg_count: int) -> float:
    """
    Calculate score for a user's message count with anti-spam measures.

    First 5 messages get full quadratic voting weight (sqrt),
    messages 6-20 get logarithmic attenuation,
    messages beyond 20 are capped (hard cutoff).

    Examples:
        1 message  = 1.00 points
        5 messages = 2.24 points
        10 messages = 2.24 + log(6) * 0.3 = 2.78 points
        20 messages = 2.24 + log(16) * 0.3 = 3.07 points
        25+ messages = 3.07 points (capped at 20)
    """
    # Cap at 20 messages max
    msg_count = min(msg_count, 20)

    if msg_count <= 5:
        return msg_count ** 0.5
    else:
        # First 5 get full sqrt weight, rest get logarithmic
        return (5 ** 0.5) + math.log(msg_count - 4) * 0.3


daily_message = """
The creation session is complete. It has been chosen as the top **Seed of the Day** and will be permanently recorded in the **Abraham Covenant**. Next, we’ll process and package everything into a final form that reflects all of the development, research, and creative work from this session.

# Structure of the Covenant

You will now produce the Covenant entry. It must include:

* A title
* A tagline
* A representative 16:9 poster image with the title on it
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

Finally, call the **`abraham_covenant`** tool to publish the blog post, poster image, and supporting content. Make sure the poster image is 16:9, regardless of the original content you made.
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
        user_messages = [m for m in messages if m.role == "user"]

        # Count messages per unique user
        messages_per_user = {}
        for msg in user_messages:
            user_id = msg.sender  # sender is an ObjectId field on ChatMessage
            if user_id:
                user_id = str(user_id)
                messages_per_user[user_id] = messages_per_user.get(user_id, 0) + 1

        # Quadratic voting with anti-spam: first 5 messages get sqrt weight,
        # messages 6-20 get logarithmic attenuation, hard cap at 20 messages.
        # Example: 4 users with 4 messages each = 8 points vs 1 user with 100 messages = 3.07 points
        score = sum(user_score(msg_count) for msg_count in messages_per_user.values())

        unique_users = len(messages_per_user)
        total_messages = len(user_messages)

        print("session", session.id)
        print("score", score)
        print("unique_users", unique_users)
        print("total_messages", total_messages)
        print("--------------------------------")

        candidates.append({
            "session": session,
            "score": score,
            "unique_users": unique_users,
            "total_messages": total_messages,
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    print("---")
    print("Session Scores (Quadratic Voting):")
    for candidate in candidates:
        print(f"  {candidate['session'].id}: score={candidate['score']:.2f}, "
              f"unique_users={candidate['unique_users']}, "
              f"total_messages={candidate['total_messages']}")
    print("---")

    winner = candidates[0]

    print("THE WINNER IS", winner["session"].id)

    if str(winner["session"].id) != "68f615d74bd332166b766ec5":
        raise Exception("Stop here, it's", str(winner["session"].id))

    # raise Exception("Stop here")


    # DB = os.getenv()


    result = await session_post.async_run({
        "role": "user",
        "session": str(winner["session"].id),
        "agent_id": str(agent.id),
        "content": daily_message,
        "attachments": [],
        # "pin": True,
        "prompt": True,
        "async": True,
        "extra_tools": ["abraham_covenant"],
    })

    return {"output": [{"session": str(winner["session"].id)}]}


async def abraham_rest(agent: Agent):
    print("RUN ABRAHAM_REST !!")
    print("agent", agent)
    print(type(agent))
    # tool = Tool.load("abraham_rest")
    # result = await tool.async_run({"agent_id": str(agent.id)}) # todo: maybe some nice Abraham comments here

    from eve.tools.abraham.abraham_rest.handler import rest

    result = rest()
    print("result rest", result)

    return {
        "output": [{
            "tx_hash": result["tx_hash"],
            "explorer_url": result["explorer_url"]
        }]
    }


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    print("RUN ABRAHAM_DAILY")
    print("agent", agent)
    print(type(agent))
    print("session", session)
    print(type(session))

    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    # check if UTC hour is 2, 4, 6, etc
    if datetime.now(timezone.utc).weekday() == 5:  # Saturday is 5 (Monday is 0)
        return await abraham_rest(agent)
    else:
        return await commit_daily_work(agent, session)


# if __name__ == "__main__":
#     agent = Agent.from_mongo("675f880479e00297cd9b4688")
#     agent_id = "675f880479e00297cd9b4688"
#     asyncio.run(handler({}, agent=agent_id))