from eve.agent.agent import Agent
from eve.tools.twitter import X
from eve.agent.session.models import Deployment


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")

    x = X(deployment)
    start_time = args.get("start_time")
    
    response = x.fetch_mentions(start_time=start_time)

    users_by_id = {
        u["id"]: u for u in response.get("includes", {}).get("users", [])
    }

    # Format the response to include relevant tweet data
    tweets = []
    for tw in response.get("data", []):
        author   = users_by_id.get(tw["author_id"], {})
        username = author.get("username", "unknown")

        tweets.append(
            {
                "id"        : tw["id"],
                "text"      : tw["text"],
                "author_id" : tw["author_id"],
                "username"  : username,
                "created_at": tw.get("created_at"),
                "url"       : f"https://x.com/{username}/status/{tw['id']}",
            }
        )

    return {"output": tweets}
