from datetime import datetime, timedelta, timezone

from bson import ObjectId

from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tool import ToolContext
from eve.tools.abraham.abraham_daily.handler import EXCLUDED_SESSION_IDS, user_score
from eve.tools.abraham.abraham_seed.handler import AbrahamCreation
from eve.tools.gigabrain.get_messages_digest.handler import (
    handler as get_messages_digest_handler,
)


async def handler(context: ToolContext):
    agent_id = context.agent
    prompt = context.args.get("prompt", "")
    hours = context.args.get("hours", 480)

    if not agent_id:
        return {"output": {"error": "agent is required"}}
    if not prompt:
        return {"output": {"error": "prompt is required"}}

    # Load the agent
    if isinstance(agent_id, str):
        agent_id = ObjectId(agent_id)
    agent = Agent.from_mongo(agent_id)
    if not agent:
        return {"output": {"error": f"Agent not found: {agent_id}"}}
    if agent.username != "abraham":
        return {"output": {"error": "This tool is only available for Abraham"}}

    # Build exclusion list: already-minted creation session_ids + manual exclusions
    creations = AbrahamCreation.find({})
    creation_session_ids = []
    for c in creations:
        if c.session_id:
            try:
                creation_session_ids.append(ObjectId(c.session_id))
            except Exception:
                pass
    excluded_ids = creation_session_ids + EXCLUDED_SESSION_IDS

    # Only consider sessions updated within the time window
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Find eligible sessions: not minted, not excluded, single-agent, within window
    sessions = Session.find(
        {
            "_id": {"$nin": excluded_ids},
            "agents": [agent.id],
            "updatedAt": {"$gte": cutoff},
        }
    )

    if not sessions:
        return {
            "output": {
                "summary": "No eligible sessions found in the specified time window.",
                "top_session_id": None,
                "top_session_title": None,
                "top_session_score": 0,
            }
        }

    # Score sessions via quadratic voting
    candidates = []
    for session in sessions:
        messages = session.get_messages()
        user_messages = [m for m in messages if m.role == "user"]

        messages_per_user = {}
        for msg in user_messages:
            user_id = msg.sender
            if user_id:
                user_id = str(user_id)
                messages_per_user[user_id] = messages_per_user.get(user_id, 0) + 1

        score = sum(user_score(msg_count) for msg_count in messages_per_user.values())

        candidates.append(
            {
                "session": session,
                "score": score,
                "total_messages": len(user_messages),
            }
        )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    winner = candidates[0]

    # Collect eligible session IDs for the digest
    eligible_session_ids = [str(c["session"].id) for c in candidates]

    # Call get_messages_digest internally with eligible sessions + the prompt
    digest_context = ToolContext(
        args={
            "session_ids": eligible_session_ids,
            "hours": hours,
            "instructions": prompt,
        },
        agent=str(agent.id),
    )
    digest_result = await get_messages_digest_handler(digest_context)
    digest_output = digest_result.get("output", {})

    return {
        "output": {
            "summary": digest_output.get("summary", ""),
            "top_session_id": str(winner["session"].id),
            "top_session_title": winner["session"].title or "",
            "top_session_score": round(winner["score"], 2),
        }
    }
