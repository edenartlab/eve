from .. import veo_handler

async def handler(args: dict, user: str = None, agent: str = None):
    return await veo_handler(
        args, 
        model="veo-2.0-generate-001"
    )