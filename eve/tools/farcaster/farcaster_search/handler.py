import os
import aiohttp
from datetime import datetime, timedelta
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.tool import ToolContext


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    # Get search parameters
    query = context.args.get("query")
    author = context.args.get("author")
    channel = context.args.get("channel")
    limit = min(context.args.get("limit", 10), 100)  # Cap at 100
    time_range_hours = context.args.get("time_range_hours")

    # Validate that at least one search parameter is provided
    if not any([query, author, channel]):
        raise Exception(
            "At least one search parameter (query, author, or channel) must be provided"
        )

    # Get Neynar API key
    neynar_api_key = os.getenv("NEYNAR_API_KEY")
    if not neynar_api_key:
        raise Exception("NEYNAR_API_KEY not found in environment")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": neynar_api_key,
                "Content-Type": "application/json",
            }

            # Use the v2 cast search endpoint as the primary method
            search_url = "https://api.neynar.com/v2/farcaster/cast/search"
            params = {"limit": limit}

            # Add query parameter if provided
            if query:
                params["q"] = query

            # Resolve author to FID if provided
            author_fid = None
            if author:
                if author.isdigit():
                    author_fid = int(author)
                else:
                    # Search for user by username
                    user_search_url = "https://api.neynar.com/v2/farcaster/user/search"
                    user_params = {"q": author, "limit": 1}

                    async with session.get(
                        user_search_url, headers=headers, params=user_params
                    ) as response:
                        if response.status == 200:
                            user_data = await response.json()
                            if user_data.get("result", {}).get("users"):
                                author_fid = user_data["result"]["users"][0]["fid"]

                        if not author_fid:
                            raise Exception(f"Could not find user: {author}")

                # Add author FID parameter
                params["author_fid"] = author_fid

            # Add channel parameter if provided
            if channel:
                params["channel_id"] = channel

            # Add time range parameters if provided
            if time_range_hours:
                start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
                params["start_time"] = start_time.isoformat() + "Z"

            # Execute the search
            async with session.get(
                search_url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result_casts = data.get("result", {}).get("casts", [])

                    results = []
                    for cast in result_casts:
                        formatted_result = format_cast_result(cast)
                        if formatted_result:
                            results.append(formatted_result)

                    return {"output": results}
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )

    except Exception as e:
        raise Exception(f"Failed to search Farcaster: {str(e)}")


def format_cast_result(cast):
    """Format a cast result from Neynar API into our standard format"""
    author = cast.get("author", {})

    return {
        "hash": cast.get("hash"),
        "text": cast.get("text", ""),
        "author_username": author.get("username"),
        "author_display_name": author.get("display_name"),
        "author_fid": author.get("fid"),
        "timestamp": cast.get("timestamp"),
        "replies_count": cast.get("replies", {}).get("count", 0),
        "likes_count": cast.get("reactions", {}).get("likes_count", 0),
        "recasts_count": cast.get("reactions", {}).get("recasts_count", 0),
        "url": f"https://warpcast.com/{author.get('username')}/{cast.get('hash')[:10]}",
        "embeds": [
            embed.get("url") for embed in cast.get("embeds", []) if embed.get("url")
        ],
        "channel": cast.get("channel", {}).get("id") if cast.get("channel") else None,
    }
