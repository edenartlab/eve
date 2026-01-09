from bson import ObjectId

from eve.agent import Agent
from eve.agent.deployments import Deployment
from eve.agent.deployments.farcaster import (
    FarcasterEvent,
    get_farcaster_user_info,
    post_cast,
)
from eve.agent.session.models import Session
from eve.tool import ToolContext

# TODO: save message id to FarcasterEvent


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    # Auto-set session platform to farcaster if not already set
    if context.session:
        session_obj = Session.from_mongo(context.session)
        if session_obj.platform is None:
            session_obj.update(platform="farcaster")

    # Get parameters from args
    text = context.args.get("text", "")
    embeds = context.args.get("embeds") or []
    parent_hash = context.args.get("reply_to")

    # Enforce reply_to after first cast in session
    if context.session and not parent_hash:
        prior_cast = FarcasterEvent.find_one({"session_id": ObjectId(context.session)})
        if prior_cast:
            raise Exception(
                "reply_to is required after your first cast. Use the cast_hash from "
                "your previous cast or the cast you're responding to. For threads, "
                "reply to your own previous cast."
            )

    # get parent FID
    parent_fid = None
    if parent_hash:
        parent_event = FarcasterEvent.find_one({"cast_hash": parent_hash})
        parent_fid = parent_event.cast_fid

    # Validate required parameters
    if not text and not embeds:
        raise Exception("Either text content or embeds must be provided")

    try:
        # Get FID - use cached value or fetch and cache on agent
        if agent.farcasterId:
            cast_fid = int(agent.farcasterId)
        else:
            user_info = await get_farcaster_user_info(deployment.secrets)
            cast_fid = int(user_info["fid"])
            agent.update(
                farcasterId=str(cast_fid), farcasterUsername=user_info.get("username")
            )

        # Prepare parent parameter if replying
        parent = None
        if parent_hash and parent_fid:
            parent = {"hash": parent_hash, "fid": parent_fid}

        outputs = []
        embeds = embeds[:4]  # limit to 4 embeds

        # break into 2 casts if there are more than 2 embeds
        if agent.username == "abraham":  # pro subscription allows 4 embeds
            embeds1, embeds2 = embeds[:4], embeds[4:]
        else:
            embeds1, embeds2 = embeds[:2], embeds[2:]

        # Post the main cast using the reusable helper function
        result = await post_cast(
            secrets=deployment.secrets, text=text, embeds=embeds1 or None, parent=parent
        )
        cast_hash = result["hash"]
        cast_url = result["url"]
        thread_hash = result.get("thread_hash")

        outputs.append({"url": cast_url, "cast_hash": cast_hash, "success": True})

        if embeds2:
            parent1 = {"hash": cast_hash, "fid": cast_fid}
            result2 = await post_cast(
                secrets=deployment.secrets,
                text="",
                embeds=embeds2,
                parent=parent1,
                thread_hash=thread_hash,
            )
            cast_hash2 = result2["hash"]
            cast_url2 = result2["url"]
            outputs.append({"url": cast_url2, "cast_hash": cast_hash2, "success": True})

        # update session key to the hash
        if context.session and outputs:
            session = Session.from_mongo(context.session)
            cast_hash = outputs[0].get("cast_hash")
            session.update(session_key=f"FC-{thread_hash}")
            # NEXT TRY: shouldn't this be using thread_hash

        # save casts as farcaster events
        for output in outputs:
            event = FarcasterEvent(
                session_id=ObjectId(context.session),
                message_id=ObjectId(context.message),
                content=text,
                cast_hash=output.get("cast_hash"),
                cast_fid=cast_fid,
                reply_cast=parent_hash,
                reply_fid=parent_fid,
                status="completed",
                event=None,
            )
            event.save()

        return {"output": outputs}

    except Exception as e:
        raise Exception(f"Failed to post Farcaster cast: {str(e)}")
