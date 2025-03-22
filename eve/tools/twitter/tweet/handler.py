from ....deploy import Deployment
from ....agent import Agent
from .. import X


async def handler(args: dict, user: str = None, agent: str = None):
    print("T1")
    print(agent)
    print(user)
    print("Ggg")
    agent = Agent.from_mongo(agent)
    print("T2")
    print(agent)
    # attempt to find a valid twitter deployment
    print("T3")
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    print("T4")
    print(deployment)
    if not deployment:
        raise Exception("No valid twitter deployments found")
    print("T5")
    x = X(deployment)
    print("T6")
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
