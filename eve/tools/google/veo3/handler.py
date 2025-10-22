from eve.tool import ToolContext
from .. import veo_handler


async def handler(context: ToolContext):
    if context.args.get("fast"):
        model = "veo-3.0-fast-generate-preview"
    else:
        model = "veo-3.0-generate-preview"

    return await veo_handler(context.args, model=model)
