import asyncio
import os

import fal_client
from loguru import logger

from eve.tool import ToolContext

# Endpoints for Nano Banana
TXT2IMG_ENDPOINT = "fal-ai/nano-banana"
IMG2IMG_ENDPOINT = "fal-ai/nano-banana/edit"

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    error_str = str(error).lower()

    # Rate limit errors (429)
    if (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
    ):
        return True

    # Server errors (5xx)
    if any(f"{code}" in error_str for code in range(500, 600)):
        return True

    # Network/timeout errors
    if any(
        term in error_str
        for term in ["timeout", "connection", "network", "unavailable"]
    ):
        return True

    return False


def _format_error_for_user(error: Exception) -> str:
    """Format error message for user-friendly display."""
    error_str = str(error).lower()

    if "429" in error_str or "rate limit" in error_str:
        return "Rate limit reached for this image model. Please try again later or use a different image model."

    if (
        "401" in error_str
        or "unauthorized" in error_str
        or "authentication" in error_str
    ):
        return "Authentication error with FAL API. Please check API credentials."

    if "403" in error_str or "forbidden" in error_str:
        return "Access denied to FAL API. Please check API permissions."

    if any(f"{code}" in error_str for code in range(500, 600)):
        return "FAL API server error. Please try again later."

    if "timeout" in error_str:
        return "Request timed out. Please try again."

    # Return original error for unknown cases
    return str(error)


async def call_fal_with_retry(endpoint: str, args: dict) -> dict:
    """
    Call FAL API with exponential backoff retry logic.

    Args:
        endpoint: The FAL API endpoint
        args: Arguments for the API call

    Returns:
        The API response dict

    Raises:
        ValueError: With user-friendly error message
    """
    delay = INITIAL_DELAY
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:

            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        logger.info(log["message"])

            result = await asyncio.to_thread(
                fal_client.subscribe,
                endpoint,
                arguments=args,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            return result

        except Exception as e:
            last_error = e
            logger.warning(
                f"FAL API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )

            # Check if error is retryable and we have retries left
            if _is_retryable_error(e) and attempt < MAX_RETRIES:
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
                continue

            # Non-retryable or max retries reached
            raise ValueError(_format_error_for_user(e))

    # Should not reach here, but just in case
    raise ValueError(_format_error_for_user(last_error))


async def handler(context: ToolContext):
    """
    Handler for Nano Banana image generation via FAL.

    Dynamically switches between txt2img and img2img endpoints based on
    whether image_urls is provided.
    """
    # Check FAL API key
    if not os.getenv("FAL_KEY"):
        raise ValueError("FAL_KEY is not set")

    args = context.args

    # Validate input
    if not args.get("prompt"):
        raise ValueError("'prompt' is required")

    # Determine endpoint based on image_urls presence
    image_urls = args.get("image_urls")
    endpoint = IMG2IMG_ENDPOINT if image_urls else TXT2IMG_ENDPOINT

    logger.info(f"Using endpoint: {endpoint}")

    # Build FAL arguments
    fal_args = {
        "prompt": args["prompt"],
        "num_images": args.get("num_images", 1),
        "aspect_ratio": args.get("aspect_ratio", "1:1"),
        "output_format": args.get("output_format", "png"),
    }

    # Add seed if provided
    if args.get("seed") is not None:
        fal_args["seed"] = args["seed"]

    # Add image_urls for img2img mode
    if image_urls:
        fal_args["image_urls"] = image_urls

    # Make the API call with retry logic
    result = await call_fal_with_retry(endpoint, fal_args)

    # Extract output URLs from result
    output_urls = []
    if "images" in result and isinstance(result["images"], list):
        for item in result["images"]:
            if isinstance(item, dict) and "url" in item:
                output_urls.append(item["url"])

    if not output_urls:
        logger.error(f"No images in FAL response: {result}")
        raise ValueError("No images were generated")

    return {"output": output_urls}
