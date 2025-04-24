import base64
import os
from openai import AsyncOpenAI

async def handler(args: dict, user: str = None, agent: str = None):
    """
    Handles the image generation request using the OpenAI API (gpt-image-1 only).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = AsyncOpenAI(api_key=api_key)

    # Prepare arguments for the OpenAI API call
    # Filter out any None values before sending
    valid_args = {k: v for k, v in args.items() if v is not None}

    # Hardcode some params:
    valid_args["model"] = "gpt-image-1"
    valid_args["moderation"] = "low" 

    if user and 'user' not in valid_args:
         valid_args['user'] = user

    try:
        print(f"Calling OpenAI Images API (gpt-image-1) with args: {valid_args}")
        response = await client.images.generate(**valid_args)

        output = []
        for item in response.data:
            image_bytes = base64.b64decode(item.b64_json)
            with open("image.jpg", "wb") as f:
                f.write(image_bytes)
            output.append("image.jpg")

        print("OUTPUT IS:", output)
        return {"output": output}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise e 