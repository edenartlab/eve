from eve.deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict):
    agent = Agent.from_mongo(args["agent"])
    # attempt to find a valid twitter deployment
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    print("DEPLOYMENT", deployment)
    if not deployment:
        raise Exception("No valid twitter deployments found")
    x = X(deployment)
    if args.get("images"):
        media_ids = [x.tweet_media(image) for image in args.get("images", [])]
        response = x.post(text=args.get("content") or "", media_ids=media_ids)
    else:
        response = x.post(text=args.get("content"))
    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{agent.username}/status/{tweet_id}"
    return {"output": url}
