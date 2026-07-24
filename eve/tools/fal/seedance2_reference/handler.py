from eve.tool import ToolContext
from eve.tools.fal.nano_banana_2_fal.handler import call_fal_with_retry

ENDPOINT = "bytedance/seedance-2.0/reference-to-video"
MAX_TOTAL_FILES = 12


async def handler(context: ToolContext):
    args = context.args
    images = args.get("reference_images") or []
    videos = args.get("reference_videos") or []
    audio = args.get("reference_audio") or []

    if not (images or videos):
        raise ValueError(
            "Provide at least one reference image or video (audio-only is not allowed)."
        )
    if len(images) + len(videos) + len(audio) > MAX_TOTAL_FILES:
        raise ValueError(
            f"Too many reference files: max {MAX_TOTAL_FILES} total across images, videos, and audio."
        )

    payload = {
        "prompt": args["prompt"],
        "resolution": args.get("resolution") or "720p",
        "duration": str(args.get("duration") or "auto"),
        "aspect_ratio": args.get("aspect_ratio") or "auto",
        "generate_audio": bool(args.get("generate_audio", True)),
        "bitrate_mode": args.get("bitrate_mode") or "standard",
    }
    if images:
        payload["image_urls"] = images
    if videos:
        payload["video_urls"] = videos
    if audio:
        payload["audio_urls"] = audio
    if args.get("seed") is not None:
        payload["seed"] = args["seed"]

    result = await call_fal_with_retry(ENDPOINT, payload)
    video = (result or {}).get("video") or {}
    url = video.get("url")
    if not url:
        raise ValueError(f"Seedance 2.0 Reference returned no video: {result}")
    return {"output": [url]}
