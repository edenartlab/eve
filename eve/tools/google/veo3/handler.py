from .. import veo_handler

async def handler(args: dict, user: str = None, agent: str = None):
    if "image" in args:
        print("Warning: input image is not supported for veo3 yet")
        args.pop("image", None)
    
    return await veo_handler(
        args, 
        model="veo-3.0-generate-preview"
    )