from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

T2I_ENDPOINT = "openai/gpt-image-2"
EDIT_ENDPOINT = "openai/gpt-image-2/image-to-image"

# yaml-size preset -> fal image_size value
SIZE_MAP = {
    "1024": "square_hd",
    "portrait": "portrait_4_3",
    "landscape": "landscape_4_3",
    "4k": {"width": 3840, "height": 2160},
}


async def handler(context: ToolContext):
    args = context.args
    input_images = args.get("input_images") or []

    payload = {
        "prompt": args["prompt"],
        "quality": args.get("quality") or "medium",
        "num_images": int(args.get("n_samples") or 1),
        "image_size": SIZE_MAP.get(args.get("image_size") or "1024", "square_hd"),
        "output_format": args.get("output_format") or "png",
    }

    if input_images:
        payload["image_urls"] = input_images
        if args.get("mask"):
            if len(input_images) != 1:
                raise ValueError("mask requires exactly one input image")
            payload["mask_url"] = args["mask"]
        endpoint = EDIT_ENDPOINT
    else:
        if args.get("mask"):
            raise ValueError("mask requires an input image")
        endpoint = T2I_ENDPOINT

    result = await call_fal_with_retry(endpoint, payload)
    images = (result or {}).get("images") or []
    urls = [im.get("url") for im in images if im.get("url")]
    if not urls:
        raise ValueError(f"GPT Image 2 returned no images: {result}")
    return {"output": urls}
