from ....deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")

    x = X(deployment)
    params = {
        "query": args["query"],
        "start_time": args.get("start_time"),
        "end_time": args.get("end_time"),
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    response = x._make_request(
        "get",
        "https://api.twitter.com/2/tweets/search/recent",
        oauth=False,
        headers={"Authorization": f"Bearer {x.bearer_token}"},
        params=params,
    )

    return {"output": response.json()}
