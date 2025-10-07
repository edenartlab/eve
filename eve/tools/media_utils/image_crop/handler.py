import os
from PIL import Image
# from ... import utils


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    from .... import utils
    
    image_url = args.get("image")

    image_filename = image_url.split("/")[-1]
    image = utils.download_file(image_url, image_filename)
    
    image = Image.open(image)
    width, height = image.size
    
    left, right = width * args.get("left"), width * (1.0 - args.get("right"))
    top, bottom = height * args.get("top"), height * (1.0 - args.get("bottom"))

    image_edited_filename = f"{image_filename}_crop{left}_{right}_{top}_{bottom}.png"
    if not os.path.exists(image_edited_filename):
        image = image.crop((int(left), int(top), int(right-left), int(bottom-top)))
        image.save(image_edited_filename)

    return {
        "output": image_edited_filename
    }
    