import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import call_service


async def handler(context: ToolContext):
    entity_id = context.args["entity_id"]
    action = context.args["action"]

    # Determine the domain from entity_id
    domain = entity_id.split(".")[0]

    # Map common domains to their service domains
    service_domain_map = {
        "switch": "switch",
        "light": "light",
        "fan": "fan",
        "cover": "cover",
        "input_boolean": "input_boolean",
        "automation": "automation",
        "scene": "scene",
        "script": "script",
    }

    # Use homeassistant domain as fallback for generic turn_on/turn_off/toggle
    service_domain = service_domain_map.get(domain, "homeassistant")

    result = await call_service(
        domain=service_domain,
        service=action,
        service_data={"entity_id": entity_id},
    )

    return {"output": json.dumps({"success": True, "result": result}, indent=2)}
