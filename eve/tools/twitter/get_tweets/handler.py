from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, requester: str = None):
    agent = Agent.load(args["agent"])
    x = X(agent)

    mentions = x.fetch_mentions()
    return mentions
