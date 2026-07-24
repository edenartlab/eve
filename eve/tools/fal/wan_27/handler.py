from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

T2V_ENDPOINT = "fal-ai/wan/v2.7/text-to-video"
I2V_ENDPOINT = "fal-ai/wan/v2.7/image-to-video"


async def handler(context: ToolContext):
    args = context.args
    start_image = args.get("start_image")

    payload = {
        "prompt": args["prompt"],
        "resolution": args.get("resolution") or "1080p",
        "duration": int(args.get("duration") or 5),
        "aspect_ratio": args.get("aspect_ratio") or "16:9",
    }
    if args.get("audio_reference"):
        payload["audio_url"] = args["audio_reference"]
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    if start_image:
        payload["image_url"] = start_image
        if args.get("end_image"):
            payload["end_image_url"] = args["end_image"]
        endpoint = I2V_ENDPOINT
    else:
        if args.get("end_image"):
            raise ValueError("end_image requires start_image")
        endpoint = T2V_ENDPOINT

    result = await call_fal_with_retry(endpoint, payload)
    video = (result or {}).get("video") or {}
    url = video.get("url")
    if not url:
        raise ValueError(f"Wan 2.7 returned no video: {result}")
    return {"output": [url]}
