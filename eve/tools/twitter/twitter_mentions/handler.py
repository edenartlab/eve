from eve.user import User
from ....deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")

    x = X(deployment)
    start_time = args.get("start_time")
    response = x.fetch_mentions(start_time=start_time)

    # Format the response to include relevant tweet data
    tweets = []
    print("Tweet mentions")
    for tweet in response.get("data", []):
        print("--------------------------------")
        print(tweet["id"], tweet["text"])
        print(tweet.keys())
        print(tweet)
        tweets.append(
            {
                "id": tweet["id"],
                "text": tweet["text"],
                "author_id": tweet["author_id"],
                "created_at": tweet.get("created_at"),
                "url": f"https://x.com/{tweet['username']}/status/{tweet['id']}",
            }
        )

    return {"output": tweets}
