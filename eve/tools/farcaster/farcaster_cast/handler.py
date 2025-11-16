from typing import Any, Dict, Literal, Optional

from bson import ObjectId

from eve.agent import Agent
from eve.agent.deployments import Deployment
from eve.agent.deployments.farcaster import get_fid, post_cast
from eve.agent.session.models import Session
from eve.mongo import Collection, Document
from eve.tool import ToolContext


@Collection("farcaster_events")
class FarcasterEvent(Document):
    cast_hash: str
    event: Optional[Dict[str, Any]] = None
    status: Literal["running", "completed", "failed"]
    error: Optional[str] = None
    session_id: Optional[ObjectId] = None
    message_id: Optional[ObjectId] = None
    reply_cast: Optional[Dict[str, Any]] = None
    reply_fid: Optional[int] = None


# TODO: save message id to FarcasterEvent


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    # Get parameters from args
    text = context.args.get("text", "")
    embeds = context.args.get("embeds") or []
    parent_hash = context.args.get("parent_hash")
    parent_fid = context.args.get("parent_fid")

    # Validate required parameters
    if not text and not embeds:
        raise Exception("Either text content or embeds must be provided")

    try:
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
            # Get FID for the parent parameter
            fid = await get_fid(deployment.secrets)
            parent1 = {"hash": cast_hash, "fid": int(fid)}
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

        TARGET_FID = agent.farcasterId

        # save casts as farcaster events
        for output in outputs:
            event = FarcasterEvent(
                session_id=ObjectId(context.session),
                # message_id=new_messages[0].id,
                cast_hash=output.get("cast_hash"),
                reply_cast=output,
                reply_fid=TARGET_FID,
                status="completed",
                event=None,
            )
            event.save()

        return {"output": outputs}

    except Exception as e:
        raise Exception(f"Failed to post Farcaster cast: {str(e)}")
