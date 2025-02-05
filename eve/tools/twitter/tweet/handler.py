from ....agent import Agent
from .. import X


async def handler(args: dict):
    agent = Agent.load(args["agent"])
    x = X(agent)
    if args["images"]:
        media_ids = [x.tweet_media(image) for image in args["images"]]
        response = x.post(tweet_text=args["content"], media_ids=media_ids)
    else:
        response = x.post(tweet_text=args["content"])
    url = response.get("data", {}).get("url")
    return {"url": url}
