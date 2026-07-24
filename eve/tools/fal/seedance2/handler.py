

from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry  # shared retry

T2V_ENDPOINT = "bytedance/seedance-2.0/text-to-video"
I2V_ENDPOINT = "bytedance/seedance-2.0/image-to-video"


async def handler(context: ToolContext):
    """Seedance 2.0: routes to the t2v or i2v endpoint based on start_image."""
    args = context.args
    start_image = args.get("start_image")

    payload = {
        "prompt": args["prompt"],
        "resolution": args.get("resolution") or "720p",
        "duration": str(args.get("duration") or "auto"),
        "aspect_ratio": args.get("aspect_ratio") or "auto",
        "generate_audio": bool(args.get("generate_audio", True)),
        "bitrate_mode": args.get("bitrate_mode") or "standard",
    }
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    if start_image:
        payload["image_url"] = start_image
        if args.get("end_image"):
            payload["end_image_url"] = args["end_image"]
        endpoint = I2V_ENDPOINT
    else:
        if args.get("end_image"):
            raise ValueError("end_image requires start_image (first/last frame pair)")
        endpoint = T2V_ENDPOINT

    result = await call_fal_with_retry(endpoint, payload)
    video = (result or {}).get("video") or {}
    url = video.get("url")
    if not url:
        raise ValueError(f"Seedance 2.0 returned no video: {result}")
    return {"output": [url]}
