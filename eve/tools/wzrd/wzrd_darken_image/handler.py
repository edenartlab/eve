import tempfile
from pathlib import Path

from PIL import Image

from eve.tool import ToolContext
from eve.utils import download_file


async def handler(context: ToolContext):
    """
    Darken an image for projection mapping.

    Applies gamma compression and brightness scaling to prepare images
    for VJ projection applications.
    """
    from wzrd import darken_image

    # Get parameters
    image_url = context.args["image"]
    gamma = context.args.get("gamma", 1.5)
    max_brightness = context.args.get("max_brightness", 0.15)

    # Download the input image
    image_path = download_file(image_url, "input_image.png")
    image = Image.open(image_path)

    # Apply darkening
    darkened_arr = darken_image(image, gamma=gamma, max_brightness=max_brightness)

    # Convert result to PIL Image and save
    darkened_image = Image.fromarray(darkened_arr)

    # Save to temp file
    output_path = Path(tempfile.gettempdir()) / "darkened_output.png"
    darkened_image.save(output_path)

    return {
        "output": str(output_path),
        "intermediate_outputs": {
            "gamma": gamma,
            "max_brightness": max_brightness,
        },
    }
