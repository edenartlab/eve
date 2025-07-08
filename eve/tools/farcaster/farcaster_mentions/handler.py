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

    # Get parameters
    username = args.get("username")
    additional_query = args.get("additional_query")
    limit = min(args.get("limit", 10), 100)  # Cap at 100
    time_range_hours = args.get("time_range_hours", 24)  # Default 24 hours

    # Get Neynar API key
    neynar_api_key = os.getenv("NEYNAR_API_KEY")
    if not neynar_api_key:
        raise Exception("NEYNAR_API_KEY not found in environment")

    try:
        from farcaster import Warpcast

        # Initialize Farcaster client to get agent's own info if needed
        client = Warpcast(mnemonic=deployment.secrets.farcaster.mnemonic)

        target_username = username
        target_fid = None

        # If no username provided, use the agent's own account
        if not target_username:
            user_info = client.get_me()
            target_username = user_info.username
            target_fid = user_info.fid

        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": neynar_api_key,
                "Content-Type": "application/json",
            }

            # If we don't have the FID yet, resolve username to FID
            if not target_fid:
                if target_username.isdigit():
                    target_fid = int(target_username)
                else:
                    # Search for user by username
                    user_search_url = f"https://api.neynar.com/v2/farcaster/user/search"
                    user_params = {"q": target_username, "limit": 1}

                    async with session.get(
                        user_search_url, headers=headers, params=user_params
                    ) as response:
                        if response.status == 200:
                            user_data = await response.json()
                            if user_data.get("result", {}).get("users"):
                                target_fid = user_data["result"]["users"][0]["fid"]
                                target_username = user_data["result"]["users"][0][
                                    "username"
                                ]

                        if not target_fid:
                            raise Exception(f"Could not find user: {target_username}")

            # Search for mentions using Neynar's mentions endpoint
            mentions_url = f"https://api.neynar.com/v2/farcaster/mentions-and-replies"
            params = {
                "fid": target_fid,
                "limit": limit,
            }

            if time_range_hours:
                start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
                params["start_time"] = start_time.isoformat() + "Z"

            results = []
            async with session.get(
                mentions_url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    notifications = data.get("notifications", [])

                    for notification in notifications:
                        cast = notification.get("cast")
                        if not cast:
                            continue

                        # Filter by additional query if provided
                        cast_text = cast.get("text", "").lower()
                        if (
                            additional_query
                            and additional_query.lower() not in cast_text
                        ):
                            continue

                        # Check if this is actually a mention (not just a reply)
                        mentioned_profiles = cast.get("mentioned_profiles", [])
                        is_mention = any(
                            profile.get("fid") == target_fid
                            for profile in mentioned_profiles
                        )

                        if is_mention or f"@{target_username}" in cast_text:
                            results.append(
                                format_mention_result(
                                    cast, notification, target_username
                                )
                            )

                elif response.status == 404:
                    # Try alternative approach using search
                    search_query = f"@{target_username}"
                    if additional_query:
                        search_query += f" {additional_query}"

                    search_url = "https://api.neynar.com/v2/farcaster/cast/search"
                    search_params = {
                        "q": search_query,
                        "limit": limit,
                    }

                    if time_range_hours:
                        start_time = datetime.utcnow() - timedelta(
                            hours=time_range_hours
                        )
                        search_params["start_time"] = start_time.isoformat() + "Z"

                    async with session.get(
                        search_url, headers=headers, params=search_params
                    ) as search_response:
                        if search_response.status == 200:
                            search_data = await search_response.json()
                            for cast in search_data.get("casts", []):
                                # Verify this is actually a mention
                                cast_text = cast.get("text", "").lower()
                                if f"@{target_username.lower()}" in cast_text:
                                    results.append(
                                        format_mention_result(
                                            cast, None, target_username
                                        )
                                    )
                else:
                    raise Exception(f"API request failed with status {response.status}")

            return {"output": results}

    except Exception as e:
        raise Exception(f"Failed to find Farcaster mentions: {str(e)}")


def format_mention_result(cast, notification=None, target_username=None):
    """Format a mention result from Neynar API into our standard format"""
    author = cast.get("author", {})

    result = {
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
        "mentioned_user": target_username,
    }

    # Add notification type if available
    if notification:
        result["notification_type"] = notification.get("type")

    return result
