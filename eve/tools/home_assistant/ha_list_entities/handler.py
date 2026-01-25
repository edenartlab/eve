import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_all_states


async def handler(context: ToolContext):
    domain = context.args.get("domain")
    states = await get_all_states()

    if domain:
        states = [s for s in states if s["entity_id"].startswith(f"{domain}.")]

    # Return a summary: entity_id, state, and friendly_name
    summary = [
        {
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
        }
        for s in states
    ]

    return {"output": json.dumps(summary, indent=2)}
