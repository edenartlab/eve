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
    post = args.get("post")


    abraham_creation = AbrahamCreation.from_mongo(session)

    print("THIS IS THE ABRAHAM CREATION")
    print(abraham_creation)

    print("title")
    print(title)
    print("post")
    print(post)

    return {"output": [{"session": session}]}

