import asyncio
import mimetypes
import tempfile
from typing import Any, Dict, List
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


async def build_content_parts(message: Dict[str, Any]) -> List[Any]:
    """Build content parts from a message dict."""
    parts = []

    # Add text content if present
    if message.get("content"):
        parts.append(genai.types.Part(text=message["content"]))

    # Add image attachments if present
    if message.get("attachments"):
        for attachment_url in message["attachments"]:
            image_bytes, mime_type = await asyncio.to_thread(
                download_image, attachment_url
            )
            parts.append(
                genai.types.Part(
                    inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes)
                )
            )

    return parts


async def handler(context: ToolContext):
    """
    Handler for Gemini 3 Pro Image generation.

    Supports multi-turn conversations with image inputs and generates images.
    """
    args = context.args

    # Validate input
    if not args.get("messages"):
        raise ValueError("'messages' is required")

    # Create GCP client. Gemini 3 Pro Image requires global
    client = create_gcp_client(gcp_location="global")

    # Build contents for the API call
    contents = []

    for message in args["messages"]:
        role = message.get("role", "user")
        parts = await build_content_parts(message)

        if parts:
            contents.append(genai.types.Content(role=role, parts=parts))

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

    generation_config = genai.types.GenerateContentConfig(**config_dict)

    # Make the API call
    response = await asyncio.to_thread(
        client.models.generate_content,
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
