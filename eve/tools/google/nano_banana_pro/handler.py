import asyncio
import mimetypes
import os
import re
import tempfile
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import errors as genai_errors
from loguru import logger

from eve.tool import ToolContext
from eve.tools.google import create_gcp_client


async def generate_content_with_retry(
    client,
    model: str,
    contents: list,
    config: genai.types.GenerateContentConfig,
    max_retries: int = 3,
    initial_delay: float = 1.0,
):
    """
    Make a generate_content call with exponential backoff retry logic for rate limits.

    Args:
        client: The Google GenAI client
        model: Model name to use
        contents: Content to generate from
        config: Generation configuration
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)

    Returns:
        The API response

    Raises:
        ValueError: With user-friendly error message
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response

        except genai_errors.ClientError as e:
            logger.error(
                f"Google API ClientError in nano_banana_pro: code={e.code}, message={e.message}, details={e.details}, attempt={attempt+1}/{max_retries+1}"
            )
            # Handle rate limiting (429) with retry
            if e.code == 429 and attempt < max_retries:
                # Exponential backoff: wait 1s, 2s, 4s, etc.
                await asyncio.sleep(delay)
                delay *= 2
                continue
            elif e.code == 429:
                raise ValueError(
                    "Rate limit reached for this image model. Please try again later or use a different image model (e.g., model_preference='flux' or 'openai')."
                )
            elif e.code == 403:
                raise ValueError(
                    "Google API access denied. Please check your API credentials and quotas."
                )
            elif e.code == 400:
                raise ValueError(f"Invalid request to Google API: {e.message}")
            else:
                raise ValueError(f"Google API error ({e.code}): {e.message}")

        except genai_errors.ServerError as e:
            logger.error(
                f"Google API ServerError in nano_banana_pro: code={e.code}, message={e.message}, details={e.details}, attempt={attempt+1}/{max_retries+1}"
            )
            # Retry server errors (5xx) with backoff
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            else:
                raise ValueError(
                    f"Google API server error. Please try again later or use a different image model. ({e.code})"
                )

        except Exception as e:
            # Don't retry unexpected errors
            logger.error(
                f"Unexpected error in nano_banana_pro: {type(e).__name__}: {str(e)}"
            )
            raise ValueError(f"Unexpected error calling Google API: {str(e)}")

    raise ValueError("Maximum retries exceeded for Google API call")


async def download_image(url: str) -> tuple[bytes, str]:
    """Download an image from a URL and return its bytes and MIME type."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
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
            image_bytes, mime_type = await download_image(image_url)
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
    allowed_aspect_ratios = {
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "9:16",
        "16:9",
        "21:9",
    }
    image_config_dict = {}
    aspect_ratio = args.get("aspect_ratio")
    if aspect_ratio:
        if aspect_ratio == "match_input_image":
            # Gemini API does not accept this literal value; let the model decide.
            pass
        elif aspect_ratio in allowed_aspect_ratios:
            image_config_dict["aspect_ratio"] = aspect_ratio
        else:
            raise ValueError(
                f"Invalid aspect_ratio '{aspect_ratio}'. Supported values: "
                + ", ".join(sorted(allowed_aspect_ratios))
            )
    if args.get("image_size"):
        image_config_dict["image_size"] = args["image_size"]
    if image_config_dict:
        config_dict["image_config"] = genai.types.ImageConfig(**image_config_dict)

    generation_config = genai.types.GenerateContentConfig(**config_dict)

    if os.getenv("EDEN_NANO_BANANA_DEBUG") == "true":
        parts_summary = []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                parts_summary.append(
                    {
                        "type": "image",
                        "mime_type": getattr(inline, "mime_type", None),
                        "bytes": len(getattr(inline, "data", b"")),
                    }
                )
            else:
                text = getattr(part, "text", None)
                parts_summary.append(
                    {
                        "type": "text",
                        "length": len(text or ""),
                        "preview": (text or "")[:200],
                    }
                )
        logger.info(
            "[NANO_BANANA_DEBUG] request",
            {
                "model": "gemini-3-pro-image-preview",
                "config": config_dict,
                "parts": parts_summary,
            },
        )

    # Make the API call with automatic retry and exponential backoff
    response = await generate_content_with_retry(
        client=client,
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=generation_config,
        max_retries=1,
        initial_delay=10.0,
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
        # Extract error details from candidates if available
        error_details = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.finish_message:
                    error_details.append(candidate.finish_message)
                elif candidate.finish_reason:
                    error_details.append(str(candidate.finish_reason))

        error_msg = (
            "; ".join(error_details) if error_details else "No images were generated"
        )
        # Filter out Google support codes (not useful to end users)
        error_msg = re.sub(r"\s*Support code: \d+\.?", "", error_msg).strip()
        logger.error(f"No images were generated: {response}")
        raise ValueError(error_msg)

    result = {"output": output_images}
    if output_text:
        result["text"] = "\n".join(output_text)

    return result
