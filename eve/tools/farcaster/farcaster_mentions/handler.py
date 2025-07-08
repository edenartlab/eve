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
                    # Search for user by username using v2 API
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

            # Use Neynar Hub API for mentions
            mentions_url = "https://hub-api.neynar.com/v1/castsByMention"
            params = {
                "fid": target_fid,
                "pageSize": limit,
                "reverse": "true",  # Get latest mentions first (must be string)
            }

            print(
                f"***debug*** Searching mentions for FID {target_fid} (username: {target_username})"
            )

            results = []
            async with session.get(
                mentions_url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(
                        f"***debug*** Found {len(data.get('messages', []))} mention messages"
                    )

                    messages = data.get("messages", [])

                    # Filter by time range if specified (skip if 0 or None)
                    if time_range_hours and time_range_hours > 0:
                        # Farcaster epoch: January 1, 2021 00:00:00 UTC
                        FARCASTER_EPOCH = 1609459200

                        cutoff_unix_timestamp = int(
                            (
                                datetime.utcnow() - timedelta(hours=time_range_hours)
                            ).timestamp()
                        )
                        # Convert to Farcaster timestamp for comparison
                        cutoff_farcaster_timestamp = (
                            cutoff_unix_timestamp - FARCASTER_EPOCH
                        )

                        print(
                            f"***debug*** Cutoff Unix timestamp: {cutoff_unix_timestamp}"
                        )
                        print(
                            f"***debug*** Cutoff Farcaster timestamp: {cutoff_farcaster_timestamp}"
                        )

                        # Debug: show some sample timestamps
                        if messages:
                            sample_timestamps = [
                                msg.get("data", {}).get("timestamp", 0)
                                for msg in messages[:3]
                            ]
                            print(
                                f"***debug*** Sample Farcaster timestamps: {sample_timestamps}"
                            )

                        messages = [
                            msg
                            for msg in messages
                            if msg.get("data", {}).get("timestamp", 0)
                            >= cutoff_farcaster_timestamp
                        ]
                        print(
                            f"***debug*** After time filtering: {len(messages)} messages"
                        )

                    for message in messages:
                        cast_data = message.get("data", {}).get("castAddBody", {})
                        if not cast_data:
                            continue

                        cast_text = cast_data.get("text", "")

                        # Filter by additional query if provided
                        if (
                            additional_query
                            and additional_query.lower() not in cast_text.lower()
                        ):
                            continue

                        # Verify this message actually mentions our target FID
                        mentions = cast_data.get("mentions", [])
                        if target_fid not in mentions:
                            continue

                        # Format the result
                        result = format_hub_mention_result(message, target_username)
                        if result:
                            results.append(result)

                else:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )

            print(f"***debug*** Returning {len(results)} mention results")
            return {"output": results}

    except Exception as e:
        raise Exception(f"Failed to find Farcaster mentions: {str(e)}")


def format_hub_mention_result(message, target_username=None):
    """Format a mention result from Neynar Hub API into our standard format"""
    try:
        data = message.get("data", {})
        cast_data = data.get("castAddBody", {})

        if not cast_data:
            return None

        # Extract basic cast info
        text = cast_data.get("text", "")
        timestamp = data.get("timestamp", 0)
        fid = data.get("fid", 0)
        cast_hash = message.get("hash", "")

        # Convert Farcaster timestamp to Unix timestamp, then to ISO format
        if isinstance(timestamp, int) and timestamp > 0:
            # Farcaster epoch: January 1, 2021 00:00:00 UTC
            FARCASTER_EPOCH = 1609459200
            unix_timestamp = timestamp + FARCASTER_EPOCH
            formatted_timestamp = (
                datetime.fromtimestamp(unix_timestamp).isoformat() + "Z"
            )
        else:
            formatted_timestamp = datetime.utcnow().isoformat() + "Z"

        # Extract embeds - Hub API embeds are structured differently
        embeds = []
        for embed in cast_data.get("embeds", []):
            if "url" in embed:
                embeds.append(embed["url"])

        # Create result with available data
        result = {
            "hash": cast_hash,
            "text": text,
            "author_fid": fid,
            "timestamp": formatted_timestamp,
            "url": f"https://warpcast.com/~/conversations/{cast_hash}",
            "embeds": embeds,
            "mentioned_user": target_username,
            "raw_farcaster_timestamp": timestamp,
            "raw_unix_timestamp": unix_timestamp
            if isinstance(timestamp, int) and timestamp > 0
            else None,
        }

        print(
            f"***debug*** Formatted mention result: {result['hash'][:10]}... from FID {fid}"
        )
        return result

    except Exception as e:
        print(f"***debug*** Error formatting mention result: {str(e)}")
        return None
