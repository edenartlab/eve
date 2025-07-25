from eve.agent.agent import Agent
from eve.tools.twitter import X
from eve.agent.session.models import Deployment


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent_obj = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")
    x = X(deployment)
    if args.get("images"):
        media_ids = [x.tweet_media(image) for image in args.get("images", [])]
        response = x.post(
            text=args.get("content") or "",
            media_ids=media_ids,
            reply=args.get("reply_to"),
        )
    elif args.get("video"):
        media_ids = [x.tweet_media(args.get("video"))]
        response = x.post(
            text=args.get("content") or "",
            media_ids=media_ids,
            reply=args.get("reply_to"),
        )
    else:
        response = x.post(text=args.get("content"), reply=args.get("reply_to"))
    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{deployment.config.twitter.username}/status/{tweet_id}"
    return {
        "output": [
            {
                "id": tweet_id,
                "url": url,
                "text": args.get("content"),
                "author_id": response.get("data", {}).get("author_id", "unknown"),
            }
        ]
    }
