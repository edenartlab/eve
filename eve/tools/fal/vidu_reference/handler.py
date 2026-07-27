from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

ENDPOINT = "fal-ai/vidu/reference-to-video"
MAX_IMAGES = 3


async def handler(context: ToolContext):
    args = context.args
    images = args.get("reference_images") or []
    if not images:
        raise ValueError("Provide at least one reference image.")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"Too many reference images: max {MAX_IMAGES}.")

    payload = {
        "prompt": args["prompt"],
        "reference_image_urls": images,
        "aspect_ratio": args.get("aspect_ratio") or "16:9",
        "movement_amplitude": args.get("movement_amplitude") or "auto",
    }
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    result = await call_fal_with_retry(ENDPOINT, payload)
    video = (result or {}).get("video") or {}
    url = video.get("url")
    if not url:
        raise ValueError(f"Vidu reference-to-video returned no video: {result}")
    return {"output": [url]}
