from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import textwrap
from pydantic import ConfigDict

import instructor
from ...eden_utils import download_file

async def handler(args: dict, user: str = None, requester: str = None):
    # print("args", args)


    class DrakepostingMeme(BaseModel):
        """In the Drakeposting meme, Drake is seen dismissively waving off or rejecting something, whereas in the bottom right, Drake is smiling and pointing as if to say "Yes," representing something he likes or agrees with."""
        
        top_disliked: str = Field(description="Text of what Drake dismisses or dislikes (2-10 words)")
        bottom_liked: str = Field(description="Text of what Drake likes or agrees with (2-10 words)")

        model_config = ConfigDict(
            json_schema_extra={
                "examples": [
                    {"top_disliked": "Waking up early to exercise.", "bottom_liked": "Sleeping in and exercising later."},
                    {"top_disliked": "Paying full price for software.", "bottom_liked": "Using open-source alternatives."},
                ]
            }
        )

    client = instructor.from_openai(OpenAI())
    meme = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=DrakepostingMeme,
        messages=[
            {
                "role": "system",
                "content": "You are a hilarious troll who comes up with iconic memes for social media. You will be given a prompt by a user and come up with a funny meme for that prompt, using the Drakeposting meme.",
            },
            {
                "role": "user",
                "content": args["prompt"],
            },
        ],
    )    

    print(meme)

    drake_image = download_file("https://imgflip.com/s/meme/Drake-Hotline-Bling.jpg", "drake.png")

    image = Image.open(drake_image)
    draw = ImageDraw.Draw(image)
    
    # Load the Impact font
    font_ttf = download_file("https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/00f1fc230ac99f9b97ba1a7c214eb5b909a78660cb3826fca7d64c3af5a14848.sfnt", "impact.ttf")
    font = ImageFont.truetype(font_ttf, size=int(image.height/10))
    
    # Define stroke width and color for better visibility
    stroke_width = int(image.height / 200)
    stroke_fill = "black"

    w, h = image.size
    box_top_right = (w/2, 0.1 * h, w/2, 0.3 * h)
    box_bottom_right = (w/2, 0.6 * h, w/2, 0.3 * h)

    # Define text positioning function
    def draw_text(draw, text, box):
        x1, y1, width, height = box
        
        # Center position is now middle of right half
        center_x = x1 + width/2
        center_y = y1 + height/2
        
        # Adjust wrap width to use full right half width
        char_width = width / (font.size / 2)
        wrapped_text = textwrap.fill(text, width=int(char_width))
        
        text_width, text_height = draw.textbbox((0, 0), wrapped_text, font=font)[2:4]
        
        # Position text in center of box
        x = center_x - text_width/2
        y = center_y - text_height/2
        
        draw.text((x, y), wrapped_text, font=font, fill="white", 
                 stroke_width=stroke_width, stroke_fill=stroke_fill)
    
    # Draw top text
    draw_text(draw, meme.top_disliked, box_top_right)
    draw_text(draw, meme.bottom_liked, box_bottom_right)
    
    # Save the modified image
    image.save("meme.jpg")

    result = {
        "output": "meme.jpg",
        "intermediate_outputs": {
            "bottom_liked": meme.bottom_liked,
            "top_disliked": meme.top_disliked,
        }
    }
    return result
