from ....deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict):
    agent = Agent.from_mongo(args.get("agent"))
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")
    x = X(deployment)
    if args.get("images"):
        media_ids = [x.tweet_media(image) for image in args.get("images", [])]
        response = x.post(text=args.get("content") or "", media_ids=media_ids)
    elif args.get("video"):
        media_ids = [x.tweet_video(args.get("video"))]
        response = x.post(text=args.get("content") or "", media_ids=media_ids)
    else:
        response = x.post(text=args.get("content"))
    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{deployment.config.twitter.username}/status/{tweet_id}"
    return {"output": url}
