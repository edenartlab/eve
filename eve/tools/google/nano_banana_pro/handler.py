from eve.tool import ToolContext

from .. import nano_banana_handler


async def handler(context: ToolContext):
    """
    Handler for Gemini 3 Pro Image generation (Nano Banana Pro).

    Takes a prompt and optional input images, generates an image.
    Uses FAL as fallback on rate limits or server errors.
    """
    return await nano_banana_handler(
        context.args,
        model="gemini-3-pro-image-preview",
        fal_fallback_tool="nano_banana_pro_fal",
    )
