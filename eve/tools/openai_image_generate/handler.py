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

        output_data = []
        for item in response.data:
            if item.b64_json: # gpt-image-1 should always have b64_json
                output_data.append({
                    "b64_json": item.b64_json,
                    "revised_prompt": item.revised_prompt
                 })
            else:
                 # This case should ideally not happen for gpt-image-1
                 print(f"Warning: Received unexpected response format from gpt-image-1: {item}")

        # If only one image requested (n=1 or n not provided), return the single item, else the list
        output = output_data[0] if len(output_data) == 1 and valid_args.get('n', 1) == 1 else output_data

        return {"output": output}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise e 