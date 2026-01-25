import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_history


async def handler(context: ToolContext):
    entity_id = context.args["entity_id"]
    start_time = context.args.get("start_time")
    end_time = context.args.get("end_time")

    result = await get_history(
        entity_id=entity_id,
        start_time=start_time,
        end_time=end_time,
    )

    return {"output": json.dumps(result, indent=2)}
