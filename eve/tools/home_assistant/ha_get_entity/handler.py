import json

from eve.tool import ToolContext
from eve.tools.home_assistant.client import get_entity


async def handler(context: ToolContext):
    entity_id = context.args["entity_id"]
    result = await get_entity(entity_id)
    return {"output": json.dumps(result, indent=2)}
