from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.load(agent)
    x = X(agent)

    mentions = x.fetch_mentions()
    return mentions
