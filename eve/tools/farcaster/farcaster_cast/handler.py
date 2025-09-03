from farcaster import Warpcast
from eve.agent.deployments import Deployment
from eve.agent import Agent


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    # Get parameters from args
    text = args.get("text", "")
    embeds = args.get("embeds") or []
    parent_hash = args.get("parent_hash")
    parent_fid = args.get("parent_fid")

    # Validate required parameters
    if not text and not embeds:
        raise Exception("Either text content or embeds must be provided")

    try:
        # Initialize Farcaster client
        client = Warpcast(mnemonic=deployment.secrets.farcaster.mnemonic)
        user_info = client.get_me()

        # Prepare parent parameter if replying
        parent = None
        if parent_hash and parent_fid:
            parent = {"hash": parent_hash, "fid": parent_fid}

        outputs = []
        embeds = embeds[:4]  # limit to 4 embeds

        # break into 2 casts if there are more than 2 embeds
        embeds1, embeds2 = embeds[:2], embeds[2:]

        # Post the main cast
        result = client.post_cast(
            text=text, embeds=embeds1 or None, parent=parent
        )
        cast_hash = result.cast.hash
        cast_url = f"https://warpcast.com/{user_info.username}/{cast_hash}"
        outputs.append({"url": cast_url, "cast_hash": cast_hash, "success": True})

        if embeds2:
            parent1 = {"hash": cast_hash, "fid": int(user_info.fid)}
            result2 = client.post_cast(
                text="", embeds=embeds2, parent=parent1
            )
            cast_hash2 = result2.cast.hash
            cast_url2 = f"https://warpcast.com/{user_info.username}/{cast_hash2}"
            outputs.append({"url": cast_url2, "cast_hash": cast_hash2, "success": True})

        return {"output": outputs}

    except Exception as e:
        raise Exception(f"Failed to post Farcaster cast: {str(e)}")
