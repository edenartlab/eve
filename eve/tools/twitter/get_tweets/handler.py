from ....agent import Agent
from .. import X


async def handler(args: dict):
    agent = Agent.load(args["agent"])
    x = X(agent)

    mentions = x.fetch_mentions()
    return mentions
