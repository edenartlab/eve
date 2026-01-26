import json
from collections import Counter

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_all_states


async def handler(context: ToolContext):
    domain = context.args["domain"]
    states = await get_all_states()

    # Filter by domain
    domain_states = [s for s in states if s["entity_id"].startswith(f"{domain}.")]

    # Count states
    state_counts = Counter(s["state"] for s in domain_states)

    # Build summary
    summary = {
        "domain": domain,
        "total_entities": len(domain_states),
        "state_counts": dict(state_counts),
        "entities": [
            {
                "entity_id": s["entity_id"],
                "state": s["state"],
                "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
            }
            for s in domain_states
        ],
    }

    return {"output": json.dumps(summary, indent=2)}
