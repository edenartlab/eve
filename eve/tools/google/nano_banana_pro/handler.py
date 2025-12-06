import mimetypes
import tempfile
from urllib.parse import urlparse

import requests
from google import genai

from eve.tool import ToolContext
from eve.tools.google import create_gcp_client


def download_image(url: str) -> tuple[bytes, str]:
    """Download an image from a URL and return its bytes and MIME type."""
    response = requests.get(url)
    response.raise_for_status()

    mime_type = response.headers.get("content-type")
    if not mime_type:
        mime_type = mimetypes.guess_type(urlparse(url).path)[0]

    return response.content, mime_type


async def handler(context: ToolContext):
    """
    Handler for Gemini 3 Pro Image generation.

    Takes a prompt and optional input images, generates an image.
    """
    args = context.args

    # Validate input
    if not args.get("prompt"):
        raise ValueError("'prompt' is required")

    # Create GCP client. Gemini 3 Pro Image requires global
    client = create_gcp_client(gcp_location="global")

    # Build content parts
    parts = []

    # Add any input images first
    if args.get("image_input"):
        for image_url in args["image_input"]:
            image_bytes, mime_type = download_image(image_url)
            parts.append(
                genai.types.Part(
                    inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)
                )
            )

    # Add the text prompt
    parts.append(genai.types.Part(text=args["prompt"]))

    # Create single user content
    contents = [genai.types.Content(role="user", parts=parts)]

    # Build generation config
    config_dict = {
        "response_modalities": ["TEXT", "IMAGE"],
    }

    # Add optional parameters
    if args.get("temperature") is not None:
        config_dict["temperature"] = args["temperature"]

    if args.get("top_p") is not None:
        config_dict["top_p"] = args["top_p"]

    if args.get("top_k") is not None:
        config_dict["top_k"] = args["top_k"]

    if args.get("max_output_tokens") is not None:
        config_dict["max_output_tokens"] = args["max_output_tokens"]

    # Image config for aspect ratio and size
    image_config_dict = {}
    if args.get("aspect_ratio"):
        image_config_dict["aspect_ratio"] = args["aspect_ratio"]
    if args.get("image_size"):
        image_config_dict["image_size"] = args["image_size"]
    if image_config_dict:
        config_dict["image_config"] = genai.types.ImageConfig(**image_config_dict)

    generation_config = genai.types.GenerateContentConfig(**config_dict)

    # Make the API call
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=generation_config,
    )

    # Extract generated images and text
    output_images = []
    output_text = []

    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data:
                        # Save the image to a temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix=".jpg", delete=False
                        ) as tmpfile:
                            tmpfile.write(part.inline_data.data)
                            output_images.append(tmpfile.name)
                    elif part.text:
                        output_text.append(part.text)

    if not output_images:
        raise ValueError("No images were generated")

    result = {"output": output_images}
    if output_text:
        result["text"] = "\n".join(output_text)

    return result
