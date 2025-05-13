from eve.agent.thread import UserMessage
from ....deploy import Deployment
from ....agent import Agent
from .. import X
from ....agent.llm import async_prompt
from pydantic import BaseModel


class TwitterSearchQuery(BaseModel):
    query: str


async def handler(args: dict):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")

    # Use async_prompt to parse the query
    system_message = """You are a Twitter search query parser. Your task is to:
1. Extract any Twitter advanced search operators from the query (e.g. FROM:, TO:, etc). Use boolean operators to combine multiple queries.
2. Return a structured query object

Twitter advanced search operators include:
- FROM:username - Tweets from a specific user
- TO:username - Tweets to a specific user
- @username - Mentions of a specific user
- min_faves:X - Tweets with minimum X likes
- min_retweets:X - Tweets with minimum X retweets
- lang:XX - Tweets in specific language
- until:YYYY-MM-DD - Tweets before date
- since:YYYY-MM-DD - Tweets after date
- filter:links - Tweets containing links
- filter:media - Tweets containing media
- filter:images - Tweets containing images
- filter:videos - Tweets containing videos

Example:
"(from:TwitterDev OR from:TwitterAPI) has:media -is:retweet"

The query should be constructed using these operators when applicable."""

    messages = [
        UserMessage(
            role="user", content=f"Parse this Twitter search query: {args['query']}"
        ),
    ]

    parsed_query = await async_prompt(
        messages=messages,
        system_message=system_message,
        response_model=TwitterSearchQuery,
        model="gpt-4o",
    )

    print("Parsed query", parsed_query)

    x = X(deployment)
    params = {
        "query": parsed_query.query,
        "start_time": args.get("start_time"),
        "end_time": args.get("end_time"),
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    response = x._make_request(
        "get",
        "https://api.twitter.com/2/tweets/search/recent",
        oauth=False,
        headers={"Authorization": f"Bearer {x.bearer_token}"},
        params=params,
    )

    print("Twitter search response", response.json())

    return {"output": response.json()}
