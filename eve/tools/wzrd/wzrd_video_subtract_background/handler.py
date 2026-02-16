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

    # Get optional parameters (let wzrd handle defaults for everything else)
    threshold = context.args.get("threshold", 10)
    gamma = context.args.get("gamma", 0.85)

    # Download input files
    video_path = download_file(video_url, "input_video.mp4", overwrite=True)
    background_path = download_file(background_url, "background.png", overwrite=True)

    # Generate output path
    output_path = Path(tempfile.gettempdir()) / "subtracted_output.mp4"

    # Process video
    result_info = subtract_background_video(
        video_path=video_path,
        background_path=background_path,
        output_path=str(output_path),
        threshold=threshold,
        gamma=gamma,
    )

    return {
        "output": str(result_info.get("output_video", output_path)),
        "intermediate_outputs": {
            "frames_processed": result_info.get("frames_processed"),
            "fps": result_info.get("fps"),
            "video_size": result_info.get("video_size"),
            "threshold": threshold,
            "gamma": gamma,
        },
    }
