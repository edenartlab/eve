from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_publish.handler import AbrahamCreation
import asyncio


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    title = args.get("title")
    tagline = args.get("tagline")
    poster_image = args.get("poster_image")
    post = args.get("post")

    abraham_creation = AbrahamCreation.find_one({
        "session_id": ObjectId(session)
    })

    print("THIS IS THE ABRAHAM CREATION")
    print(abraham_creation)

    print("title")
    print(title)
    print("post")
    print(post)

    abraham_creation.update(status="creation")

    return {"output": [{"session": session}]}

