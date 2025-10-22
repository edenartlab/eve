from eve.tool import ToolContext
from .. import veo_handler

async def handler(context: ToolContext):
    return await veo_handler(
        context.args, 
        model="veo-2.0-generate-001"
    )