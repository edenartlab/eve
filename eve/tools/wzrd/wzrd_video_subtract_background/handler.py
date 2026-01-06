import tempfile
from pathlib import Path

from eve.tool import ToolContext
from eve.utils import download_file


async def handler(context: ToolContext):
    """
    Subtract background from a video to isolate bright creatures/elements.

    Uses frame-by-frame background subtraction with streaming FFmpeg pipes
    for memory efficiency. Useful for VJ projection mapping applications.
    """
    from wzrd import subtract_background_video

    # Get required parameters
    video_url = context.args["video"]
    background_url = context.args["background"]

    # Get optional parameters with defaults
    threshold = context.args.get("threshold", 10)
    boost = context.args.get("boost", 1.1)
    feather_radius = context.args.get("feather_radius", 4)
    diff_mode = context.args.get("diff_mode", "luminance")
    output_mode = context.args.get("output_mode", "additive")
    preview = context.args.get("preview", False)
    crf = context.args.get("crf", 18)

    # Download input files
    video_path = download_file(video_url, "input_video.mp4")
    background_path = download_file(background_url, "background.png")

    # Generate output path
    output_path = Path(tempfile.gettempdir()) / "subtracted_output.mp4"
    if output_mode == "alpha":
        output_path = Path(tempfile.gettempdir()) / "subtracted_output.mov"

    # Process video
    result_info = subtract_background_video(
        video_path=video_path,
        background_path=background_path,
        output_path=str(output_path),
        threshold=threshold,
        boost=boost,
        feather_radius=feather_radius,
        diff_mode=diff_mode,
        output_mode=output_mode,
        preview=preview,
        crf=crf,
    )

    # Build output
    output = {
        "output": str(result_info.get("output_video", output_path)),
        "intermediate_outputs": {
            "frames_processed": result_info.get("frames_processed"),
            "fps": result_info.get("fps"),
            "video_size": result_info.get("video_size"),
            "threshold": threshold,
            "boost": boost,
            "diff_mode": diff_mode,
            "output_mode": output_mode,
        },
    }

    # Include preview path if generated
    if preview and result_info.get("preview_video"):
        output["intermediate_outputs"]["preview_video"] = str(
            result_info["preview_video"]
        )

    return output
