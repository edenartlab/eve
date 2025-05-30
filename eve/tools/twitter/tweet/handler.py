from eve.user import User
from ....deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, agent: str = None):
    agent_id = args.get("agent") or agent
    agent_obj = Agent.from_mongo(agent_id)
    deployment = Deployment.load(agent=agent_obj.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")
    x = X(deployment)
    if args.get("images"):
        media_ids = [x.tweet_media(image) for image in args.get("images", [])]
        response = x.post(text=args.get("content") or "", media_ids=media_ids)
    elif args.get("video"):
        media_ids = [x.tweet_media(args.get("video"))]
        response = x.post(text=args.get("content") or "", media_ids=media_ids)
    else:
        response = x.post(text=args.get("content"))
    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{deployment.config.twitter.username}/status/{tweet_id}"
    return {"output": [{
        "id": tweet_id, 
        "url": url,
        "text": args.get("content"),
        "author_id": response.get("data", {}).get("author_id", "unknown"),
    }]}
