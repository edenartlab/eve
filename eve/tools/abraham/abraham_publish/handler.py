from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")


    print("Session !!!", str(session))

    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    farcaster_post = Tool.load("farcaster_cast")

    # Get parameters from args
    text = args.get("text", "")
    media_urls = args.get("media_urls") or []

    # Validate required parameters
    if not text and not media_urls:
        raise Exception("Either text content or media URLs must be provided")

    try:
        print("Running farcaster_post")
        result = await farcaster_post.async_run({
            "agent_id": str(agent.id),
            "text": text,
            "embeds": media_urls,
        })

        print("result", result)
        #result = result["output"][0]

        return {"output": result}

    except Exception as e:
        raise Exception(f"Failed to post Farcaster cast: {str(e)}")
