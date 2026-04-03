from eve.tool import ToolContext

from .. import veo_handler


async def handler(context: ToolContext):
    if context.args.get("fast"):
        model = "veo-3.1-fast-generate-001"
    else:
        model = "veo-3.1-generate-001"

    return await veo_handler(context.args, model=model)
