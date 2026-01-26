import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_all_states


async def handler(context: ToolContext):
    query = context.args["query"].lower()
    states = await get_all_states()

    # Search in entity_id and friendly_name
    matches = []
    for s in states:
        entity_id = s["entity_id"].lower()
        friendly_name = s.get("attributes", {}).get("friendly_name", "").lower()

        if query in entity_id or query in friendly_name:
            matches.append(
                {
                    "entity_id": s["entity_id"],
                    "state": s["state"],
                    "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
                }
            )

    return {"output": json.dumps(matches, indent=2)}
