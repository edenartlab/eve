
from ....agent import Agent
from .. import X


async def handler(args: dict):
    agent = Agent.load(args["agent"])
    
    x = X(agent)

    tweet = x.post(
        tweet_text=args["content"],
        media_urls=args["images"]
    )
    print("the tweet")
    print(tweet)

    return {
        "output": tweet
    }
    