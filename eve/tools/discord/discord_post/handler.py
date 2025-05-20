from eve.user import User
from ....deploy import Deployment
from ....agent import Agent


async def handler(args: dict, user: User, agent: Agent):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")
    return {"output": []}
