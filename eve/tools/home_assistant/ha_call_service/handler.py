import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import call_service


async def handler(context: ToolContext):
    domain = context.args["domain"]
    service = context.args["service"]
    service_data = context.args.get("service_data", {})

    result = await call_service(
        domain=domain,
        service=service,
        service_data=service_data,
    )

    return {"output": json.dumps({"success": True, "result": result}, indent=2)}
