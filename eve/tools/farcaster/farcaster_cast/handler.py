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
    embeds = args.get("embeds", [])
    parent_hash = args.get("parent_hash")
    parent_fid = args.get("parent_fid")

    # Validate required parameters
    if not text and not embeds:
        raise Exception("Either text content or embeds must be provided")

    try:
        from farcaster import Warpcast

        # Initialize Farcaster client
        client = Warpcast(mnemonic=deployment.secrets.farcaster.mnemonic)

        # Prepare parent parameter if replying
        parent = None
        if parent_hash and parent_fid:
            parent = {"hash": parent_hash, "fid": parent_fid}

        # Post the cast
        result = client.post_cast(
            text=text, embeds=embeds if embeds else None, parent=parent
        )
        # Get cast URL
        cast_hash = result.cast.hash
        user_info = client.get_me()
        cast_url = f"https://warpcast.com/{user_info.username}/{cast_hash}"

        return {"output": [{"url": cast_url, "cast_hash": cast_hash, "success": True}]}

    except Exception as e:
        raise Exception(f"Failed to post Farcaster cast: {str(e)}")
