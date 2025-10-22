from eve.agent.agent import Agent
from eve.tools.twitter import X
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent_obj = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")
    x = X(deployment)
    if context.args.get("images"):
        media_ids = [x.tweet_media(image) for image in context.args.get("images", [])]
        response = x.post(
            text=context.args.get("content") or "",
            media_ids=media_ids,
            reply=context.args.get("reply_to"),
        )
    elif context.args.get("video"):
        media_ids = [x.tweet_media(context.args.get("video"))]
        response = x.post(
            text=context.args.get("content") or "",
            media_ids=media_ids,
            reply=context.args.get("reply_to"),
        )
    else:
        response = x.post(
            text=context.args.get("content"), reply=context.args.get("reply_to")
        )
    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{deployment.config.twitter.username}/status/{tweet_id}"
    return {
        "output": [
            {
                "id": tweet_id,
                "url": url,
                "text": context.args.get("content"),
                "author_id": response.get("data", {}).get("author_id", "unknown"),
            }
        ]
    }
