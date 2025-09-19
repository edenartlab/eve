from .. import veo_handler

async def handler(args: dict, user: str = None, agent: str = None):
    if args.get("fast"):
        model = "veo-3.0-fast-generate-preview"
    else:
        model = "veo-3.0-generate-preview"

    return await veo_handler(
        args, 
        model=model
    )