import os

from loguru import logger

from eve.tool import ToolContext
from eve.tools.fal_tool import call_fal

# Endpoints for Nano Banana 2
TXT2IMG_ENDPOINT = "fal-ai/nano-banana-2"
IMG2IMG_ENDPOINT = "fal-ai/nano-banana-2/edit"

# Canonical fal call: submit once, poll to completion, never resubmit.
# Kept under the historical name because several fal tools import it from here.
call_fal_with_retry = call_fal


async def handler(context: ToolContext):
    """
    Handler for Nano Banana 2 image generation via FAL.

    Dynamically switches between txt2img and img2img endpoints based on
    whether image_urls is provided. Supports resolution control and web search.
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

    # Add resolution if provided
    if args.get("resolution"):
        fal_args["resolution"] = args["resolution"]

    # Add web search if enabled
    if args.get("enable_web_search"):
        fal_args["enable_web_search"] = args["enable_web_search"]

    # Add seed if provided
    if args.get("seed") is not None:
        fal_args["seed"] = args["seed"]

    # Add image_urls for img2img mode
    if image_urls:
        fal_args["image_urls"] = image_urls
        # Relax content moderation to the permissive end for real-face edits
        # (a large share of Eden usage edits photos of real people). Gemini's
        # hard limits (minors, explicit content) still apply server-side.
        # safety_tolerance is a STRING enum "1" (strictest) .. "6" (least
        # strict); provider default is "4". Overridable if the caller sets it.
        fal_args["safety_tolerance"] = str(args.get("safety_tolerance") or "6")

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
