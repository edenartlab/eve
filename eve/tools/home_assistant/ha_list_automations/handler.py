import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_all_states


async def handler(context: ToolContext):
    states = await get_all_states()

    # Filter to automation domain
    automations = [s for s in states if s["entity_id"].startswith("automation.")]

    # Build summary
    summary = [
        {
            "entity_id": s["entity_id"],
            "state": s["state"],  # 'on' = enabled, 'off' = disabled
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
            "last_triggered": s.get("attributes", {}).get("last_triggered"),
        }
        for s in automations
    ]

    return {"output": json.dumps(summary, indent=2)}
