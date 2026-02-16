import tempfile
from pathlib import Path

from eve.tool import ToolContext
from eve.utils import download_file


async def handler(context: ToolContext):
    """
    Prepare a projection surface reference image.

    Supports two modes:
    1. Single image (darken-only): darkens a pre-cropped surface photo
    2. Two images (full pipeline): detect projection area in night image,
       align day image to it, then darken the result
    """
    from wzrd.prepare_surface import prepare_surface

    # Get parameters
    image_url = context.args["image"]
    night_image_url = context.args.get("night_image")
    max_brightness = context.args.get("max_brightness", 0.25)
    target_aspect = context.args.get("target_aspect", "16:9")
    alignment_aids = context.args.get("alignment_aids", True)

    # Download input image(s)
    image_path = download_file(image_url, "input_image.png")

    night_image_path = None
    if night_image_url:
        night_image_path = download_file(night_image_url, "night_image.png")

    # Set up output path
    output_path = Path(tempfile.gettempdir()) / "surface_output.png"

    # In full pipeline mode: image is the day photo, night_image is the night photo
    # In darken-only mode: image is the pre-cropped surface photo
    if night_image_path:
        result = prepare_surface(
            night_image_path=str(night_image_path),
            day_image_path=str(image_path),
            output_path=str(output_path),
            max_brightness=max_brightness,
            target_aspect=target_aspect,
            alignment_aids=alignment_aids,
        )
    else:
        result = prepare_surface(
            night_image_path=str(image_path),
            output_path=str(output_path),
            max_brightness=max_brightness,
            target_aspect=target_aspect,
            alignment_aids=alignment_aids,
        )

    outputs = [str(output_path)]
    if result['video']:
        outputs.append(result['video'])

    return {"output": outputs}
