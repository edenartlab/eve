import asyncio
import os

import fal_client
import httpx
from loguru import logger

from eve.tool import ToolContext

# Endpoints for Nano Banana 2
TXT2IMG_ENDPOINT = "fal-ai/nano-banana-2"
IMG2IMG_ENDPOINT = "fal-ai/nano-banana-2/edit"

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0


def _fal_status_code(error: Exception):
    """Best-effort extraction of the HTTP status code from a fal_client error.

    fal_client raises ``FalClientError(detail) from httpx.HTTPStatusError`` (see
    fal_client.client._raise_for_status), so the real status code lives on the
    chained cause's response — NOT reliably in the stringified message. Relying on
    digit-substring matching of the message misclassifies e.g. a 422 whose detail
    contains a pixel size or byte count as a "5xx server error".
    """
    for candidate in (getattr(error, "__cause__", None), error):
        resp = getattr(candidate, "response", None)
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _fal_detail(error: Exception) -> str:
    """The real fal error detail (FalClientError carries response.json()['detail'])."""
    detail = str(error).strip()
    if len(detail) > 500:
        detail = detail[:500] + "…"
    return detail


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable, preferring the real HTTP status code."""
    code = _fal_status_code(error)
    if code is not None:
        # Rate limit / transient conflicts / server errors. 4xx validation
        # errors (400/403/404/422 …) are NOT retryable — retrying wastes time
        # and hides an actionable request problem.
        return code in (408, 409, 429) or code >= 500

    # No HTTP status (transport-level failure): treat network/timeouts as retryable.
    error_str = str(error).lower()
    return any(
        term in error_str
        for term in ["timeout", "connection", "network", "unavailable", "read error"]
    )


def _format_error_for_user(error: Exception) -> str:
    """User-facing message that ALWAYS preserves fal's real status + detail.

    Keeping the detail is the whole point: a 5xx with detail "Internal Server
    Error" is a provider outage, but a 5xx/4xx whose detail is "failed to fetch
    image_urls[1]" is our own bad input — previously both were flattened to an
    identical opaque "FAL API server error", making outages indistinguishable
    from request bugs.
    """
    code = _fal_status_code(error)
    detail = _fal_detail(error)

    # Provider content-policy rejections (e.g. ByteDance's reference-to-video
    # partner validation rejects references that resemble real people — even
    # photorealistic AI-generated humans). Surface actionable guidance instead
    # of a bare 422 so an agent can adapt rather than assume an outage.
    detail_l = detail.lower()
    if "content_policy_violation" in detail_l or "likeness" in detail_l:
        return (
            "Rejected by the provider's content policy: input/reference media that "
            "resembles a real person (or contains private information) can't be "
            "processed. Use stylized, illustrated, or non-photorealistic references "
            "— or a non-human subject — and retry. "
            f"(provider detail: {detail})"
        )

    if code == 429:
        return f"Rate limit reached (FAL 429). {detail} Try again shortly or use a different model."
    if code in (401, 403):
        return f"FAL access/authentication error ({code}): {detail}"
    if code == 404:
        return f"FAL endpoint not found (404): {detail}"
    if code is not None and code >= 500:
        return f"FAL server error ({code}): {detail}. Please try again later."
    if code is not None:
        # Other 4xx — surface the real validation detail so it's actionable.
        return f"FAL rejected the request ({code}): {detail}"

    # No HTTP status: transport-level failure.
    if "timeout" in detail.lower():
        return f"FAL request timed out: {detail}"
    return detail or "FAL request failed."


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
    Handler for Nano Banana 2 image generation via FAL.

    Dynamically switches between txt2img and img2img endpoints based on
    whether image_urls is provided. Supports resolution control and web search.
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

    # Add resolution if provided
    if args.get("resolution"):
        fal_args["resolution"] = args["resolution"]

    # Add web search if enabled
    if args.get("enable_web_search"):
        fal_args["enable_web_search"] = args["enable_web_search"]

    # Add seed if provided
    if args.get("seed") is not None:
        fal_args["seed"] = args["seed"]

    # Add image_urls for img2img mode
    if image_urls:
        fal_args["image_urls"] = image_urls
        # Relax content moderation to the permissive end for real-face edits
        # (a large share of Eden usage edits photos of real people). Gemini's
        # hard limits (minors, explicit content) still apply server-side.
        # safety_tolerance is a STRING enum "1" (strictest) .. "6" (least
        # strict); provider default is "4". Overridable if the caller sets it.
        fal_args["safety_tolerance"] = str(args.get("safety_tolerance") or "6")

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
