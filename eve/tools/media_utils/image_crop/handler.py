import asyncio
import os

from PIL import Image

from eve.tool import ToolContext

# from ... import utils


def _handler_sync(context: ToolContext):
    from .... import utils

    image_url = context.args.get("image")

    image_filename = image_url.split("/")[-1]
    image = utils.download_file(image_url, image_filename)

    image = Image.open(image)
    width, height = image.size

    left, right = (
        width * context.args.get("left"),
        width * (1.0 - context.args.get("right")),
    )
    top, bottom = (
        height * context.args.get("top"),
        height * (1.0 - context.args.get("bottom")),
    )

    image_edited_filename = f"{image_filename}_crop{left}_{right}_{top}_{bottom}.png"
    if not os.path.exists(image_edited_filename):
        image = image.crop((int(left), int(top), int(right - left), int(bottom - top)))
        image.save(image_edited_filename)

    return {"output": image_edited_filename}


async def handler(context: ToolContext):
    return await asyncio.to_thread(_handler_sync, context)
