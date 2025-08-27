from .. import veo_handler

async def handler(args: dict, user: str = None, agent: str = None):
    return await veo_handler(
        args, 
        model="veo-3.0-generate-preview"
    )