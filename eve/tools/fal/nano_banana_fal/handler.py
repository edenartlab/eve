import os

from loguru import logger

from eve.tool import ToolContext
from eve.tools.fal_tool import call_fal

# Endpoints for Nano Banana
TXT2IMG_ENDPOINT = "fal-ai/nano-banana"
IMG2IMG_ENDPOINT = "fal-ai/nano-banana/edit"


async def handler(context: ToolContext):
    """
    Handler for Nano Banana image generation via FAL.

    Dynamically switches between txt2img and img2img endpoints based on
    whether image_urls is provided.
    """
    # Check FAL API key
    if not os.getenv("FAL_KEY"):
        raise ValueError("FAL_KEY is not set")

    args = context.args

    # Validate input
    if not args.get("prompt"):
        raise ValueError("'prompt' is required")

    # Determine endpoint based on image_urls presence
    image_urls = args.get("image_urls")
    endpoint = IMG2IMG_ENDPOINT if image_urls else TXT2IMG_ENDPOINT

    logger.info(f"Using endpoint: {endpoint}")

    # Build FAL arguments
    fal_args = {
        "prompt": args["prompt"],
        "num_images": args.get("num_images", 1),
        "aspect_ratio": args.get("aspect_ratio", "1:1"),
        "output_format": args.get("output_format", "png"),
    }

    # Add seed if provided
    if args.get("seed") is not None:
        fal_args["seed"] = args["seed"]

    # Add image_urls for img2img mode
    if image_urls:
        fal_args["image_urls"] = image_urls

    # One generation, one fal bill. No retry: see call_fal.
    result = await call_fal(endpoint, fal_args)

    # Extract output URLs from result
    output_urls = []
    if "images" in result and isinstance(result["images"], list):
        for item in result["images"]:
            if isinstance(item, dict) and "url" in item:
                output_urls.append(item["url"])

    if not output_urls:
        logger.error(f"No images in FAL response: {result}")
        raise ValueError("No images were generated")

    return {"output": output_urls}
