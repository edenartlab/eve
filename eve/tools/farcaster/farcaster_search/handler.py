import os
import aiohttp
from datetime import datetime, timedelta
from eve.agent.deployments import Deployment
from eve.agent import Agent


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    # Get search parameters
    query = args.get("query")
    author = args.get("author")
    channel = args.get("channel")
    limit = min(args.get("limit", 10), 100)  # Cap at 100
    time_range_hours = args.get("time_range_hours")

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

            results = []

            # If searching by author, use the user casts endpoint
            if author:
                # Try to resolve username to FID if it's not numeric
                fid = None
                if author.isdigit():
                    fid = int(author)
                else:
                    # Search for user by username
                    user_search_url = f"https://api.neynar.com/v2/farcaster/user/search"
                    user_params = {"q": author, "limit": 1}

                    async with session.get(
                        user_search_url, headers=headers, params=user_params
                    ) as response:
                        if response.status == 200:
                            user_data = await response.json()
                            if user_data.get("result", {}).get("users"):
                                fid = user_data["result"]["users"][0]["fid"]

                if fid:
                    # Get casts from specific user
                    casts_url = f"https://api.neynar.com/v2/farcaster/casts"
                    params = {
                        "fid": fid,
                        "limit": limit,
                    }

                    if time_range_hours:
                        start_time = datetime.utcnow() - timedelta(
                            hours=time_range_hours
                        )
                        params["start_time"] = start_time.isoformat() + "Z"

                    async with session.get(
                        casts_url, headers=headers, params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            for cast in data.get("casts", []):
                                # Filter by query if provided
                                if (
                                    not query
                                    or query.lower() in cast.get("text", "").lower()
                                ):
                                    results.append(format_cast_result(cast))

            # If searching by text query, use search endpoint
            elif query:
                search_url = "https://api.neynar.com/v2/farcaster/cast/search"
                params = {
                    "q": query,
                    "limit": limit,
                }

                if channel:
                    params["channel_id"] = channel

                if time_range_hours:
                    start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
                    params["start_time"] = start_time.isoformat() + "Z"

                async with session.get(
                    search_url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for cast in data.get("casts", []):
                            results.append(format_cast_result(cast))

            # If searching by channel only
            elif channel:
                channel_url = f"https://api.neynar.com/v2/farcaster/channel"
                params = {"id": channel, "type": "id"}

                async with session.get(
                    channel_url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        # Then get recent casts from this channel
                        feed_url = "https://api.neynar.com/v2/farcaster/feed/channels"
                        feed_params = {
                            "channel_ids": channel,
                            "limit": limit,
                        }

                        if time_range_hours:
                            start_time = datetime.utcnow() - timedelta(
                                hours=time_range_hours
                            )
                            feed_params["start_time"] = start_time.isoformat() + "Z"

                        async with session.get(
                            feed_url, headers=headers, params=feed_params
                        ) as feed_response:
                            if feed_response.status == 200:
                                feed_data = await feed_response.json()
                                for cast in feed_data.get("casts", []):
                                    results.append(format_cast_result(cast))

            return {"output": results}

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
