import asyncio
import mimetypes
import os
import re
import tempfile
from urllib.parse import urlparse

import httpx
from loguru import logger

from google import genai
from google.genai import errors as genai_errors
from google.oauth2 import service_account


def create_gcp_client(gcp_location: str = None):
    """Create and return a Google GenAI client configured for Vertex AI."""
    if gcp_location is None:
        gcp_location = os.environ["GCP_LOCATION"]

    service_account_info = {
        "type": os.environ["GCP_TYPE"],
        "project_id": os.environ["GCP_PROJECT_ID"],
        "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
        "private_key": os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": os.environ["GCP_CLIENT_EMAIL"],
        "client_id": os.environ["GCP_CLIENT_ID"],
        "auth_uri": os.environ["GCP_AUTH_URI"],
        "token_uri": os.environ["GCP_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["GCP_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    return genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT_ID"],
        location=gcp_location,
        credentials=credentials,
    )


async def veo_handler(args: dict, model: str):
    if not args.get("prompt") and not args.get("image"):
        raise ValueError("At least one of prompt or image is required")

    # --- setup gcp client ----
    client = create_gcp_client()

    # ---- get image and setup args ----
    config_dict = {
        "duration_seconds": args.get("duration"),  # 5-8 seconds
        "number_of_videos": args.get("n_samples"),  # 1-4
        "enhance_prompt": True,
        # enhance_prompt=args.get("prompt_enhance"),
        # person_generation="dont_allow",   # safety switch
        # storage_uri="gs://MY_BUCKET/veo/" # optional – if omitted you get bytes
    }

    if args.get("aspect_ratio"):
        config_dict["aspect_ratio"] = args.get("aspect_ratio")

    if args.get("negative_prompt"):
        config_dict["negative_prompt"] = args.get("negative_prompt")

    if args.get("generate_audio"):
        config_dict["generate_audio"] = True if args.get("generate_audio") else False

    args_dict = {
        "model": model,
        "config": genai.types.GenerateVideosConfig(**config_dict),
    }

    image = args.get("image")
    if image:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(image, timeout=30.0)
            response.raise_for_status()
            mime_type = response.headers.get("content-type")
            if not mime_type:
                mime_type = mimetypes.guess_type(urlparse(image).path)[0]
            image = genai.types.Image(image_bytes=response.content, mime_type=mime_type)
            args_dict["image"] = image

    if args.get("prompt"):
        args_dict["prompt"] = args.get("prompt")

    # ---- start generation with error handling ----
    try:
        operation = await client.aio.models.generate_videos(**args_dict)
    except genai_errors.ClientError as e:
        logger.error(
            f"Google API ClientError in veo_handler: code={e.code}, message={e.message}, details={e.details}"
        )
        if e.code == 429:
            raise ValueError(
                "Rate limit reached for this video model. Please try again later or use a different video model (e.g., model_preference='kling' or 'runway')."
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
            f"Google API ServerError in veo_handler: code={e.code}, message={e.message}, details={e.details}"
        )
        raise ValueError(
            f"Google API server error. Please try again later or use a different video model. ({e.code})"
        )
    except Exception as e:
        logger.error(f"Unexpected error in veo_handler: {type(e).__name__}: {str(e)}")
        raise ValueError(f"Unexpected error calling Google API: {str(e)}")

    # ---- poll until done ----
    max_poll_attempts = 300  # 10 minutes max (300 * 2 seconds)
    poll_attempt = 0
    while not operation.done:
        if poll_attempt >= max_poll_attempts:
            raise ValueError(
                "Video generation timed out after 10 minutes. Please try again."
            )
        await asyncio.sleep(2)
        try:
            operation = await client.aio.operations.get(operation)
        except genai_errors.ClientError as e:
            logger.error(
                f"Google API ClientError while polling: code={e.code}, message={e.message}, details={e.details}"
            )
            if e.code == 429:
                # For polling, wait a bit longer before retrying
                await asyncio.sleep(5)
                continue
            else:
                raise ValueError(f"Error polling video status: {e.message}")
        poll_attempt += 1

    if not operation.response or not operation.response.generated_videos:
        raise ValueError("No videos generated")
    # ---- download the video(s) ----
    videos = []
    for result in operation.response.generated_videos:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            tmpfile.write(result.video.video_bytes)
            videos.append(tmpfile.name)

    return {"output": videos}


# ---- Nano Banana (Image Generation) ----


def _should_fallback_to_fal(error: Exception) -> bool:
    """Determine if error should trigger FAL fallback."""
    error_str = str(error).lower()
    # Fallback on: rate limits (429), server errors (5xx), quota exceeded
    return any(
        term in error_str
        for term in [
            "rate limit",
            "429",
            "quota",
            "server error",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "overloaded",
        ]
    )


def _map_args_to_fal(args: dict, is_pro: bool = False) -> dict:
    """Map GCP arguments to FAL format."""
    fal_args = {
        "prompt": args["prompt"],
        "num_images": 1,
        "aspect_ratio": args.get("aspect_ratio", "1:1"),
        "output_format": args.get("output_format", "png"),
    }

    # Map image_input to image_urls
    if args.get("image_input"):
        fal_args["image_urls"] = args["image_input"]

    # Map Pro-specific parameters
    if is_pro:
        # Map image_size to resolution for FAL
        if args.get("image_size"):
            fal_args["resolution"] = args["image_size"]

    return fal_args


async def _nano_banana_fal_fallback(args: dict, fal_tool: str) -> dict:
    """Execute generation via FAL as fallback."""
    from eve.tools.tool_handlers import load_handler

    # Determine if this is a Pro model
    is_pro = "pro" in fal_tool.lower()

    # Map args to FAL format
    fal_args = _map_args_to_fal(args, is_pro=is_pro)

    # Create a mock context for the FAL handler
    class MockContext:
        def __init__(self, args):
            self.args = args

    handler = load_handler(fal_tool)
    return await handler(MockContext(fal_args))


async def _nano_banana_gcp(args: dict, model: str) -> dict:
    """Execute generation via GCP."""
    # Check for simulated 429 error (for testing fallback)
    if os.getenv("EDEN_SIMULATE_GCP_429") == "true":
        logger.warning("[NANO_BANANA] Simulating 429 rate limit error for testing")
        raise ValueError(
            "Rate limit 429 reached for this image model. Please try again later or use a different image model (e.g., model_preference='flux' or 'openai')."
        )

    # Validate input
    if not args.get("prompt"):
        raise ValueError("'prompt' is required")

    # Create GCP client. Gemini image models require global location
    client = create_gcp_client(gcp_location="global")

    # Build content parts
    parts = []

    # Add any input images first
    if args.get("image_input"):
        for image_url in args["image_input"]:
            async with httpx.AsyncClient() as http_client:
                # Use User-Agent header to avoid 403 errors from CloudFront/WAF
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = await http_client.get(
                    image_url, timeout=30.0, headers=headers
                )
                response.raise_for_status()
                mime_type = response.headers.get("content-type")
                if not mime_type:
                    mime_type = mimetypes.guess_type(urlparse(image_url).path)[0]
                parts.append(
                    genai.types.Part(
                        inline_data=genai.types.Blob(
                            mime_type=mime_type, data=response.content
                        )
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

    # Make the API call with retry logic
    delay = 10.0
    max_retries = 1

    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=generation_config,
            )
            break  # Success, exit retry loop

        except genai_errors.ClientError as e:
            logger.error(
                f"Google API ClientError in nano_banana: code={e.code}, message={e.message}, details={e.details}, attempt={attempt+1}/{max_retries+1}"
            )
            # Handle rate limiting (429) with retry
            if e.code == 429 and attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            elif e.code == 429:
                raise ValueError(
                    "Rate limit 429 reached for this image model. Please try again later or use a different image model (e.g., model_preference='flux' or 'openai')."
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
                f"Google API ServerError in nano_banana: code={e.code}, message={e.message}, details={e.details}, attempt={attempt+1}/{max_retries+1}"
            )
            # Retry server errors (5xx) with backoff
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            else:
                raise ValueError(
                    f"Google API server error 5xx. Please try again later or use a different image model. ({e.code})"
                )

        except Exception as e:
            # Don't retry unexpected errors
            logger.error(
                f"Unexpected error in nano_banana: {type(e).__name__}: {str(e)}"
            )
            raise ValueError(f"Unexpected error calling Google API: {str(e)}")

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


async def nano_banana_handler(
    args: dict, model: str, fal_fallback_tool: str = None
) -> dict:
    """
    Shared handler for Nano Banana models with FAL fallback.

    Args:
        args: Tool arguments (prompt, image_input, aspect_ratio, etc.)
        model: GCP model name (gemini-2.5-flash-image-preview or gemini-3-pro-image-preview)
        fal_fallback_tool: Name of FAL tool to use as fallback (nano_banana_fal or nano_banana_pro_fal)

    Returns:
        dict with 'output' containing list of image file paths
    """
    # Try GCP first
    try:
        logger.info(f"[NANO_BANANA] Attempting GCP generation with model={model}")
        result = await _nano_banana_gcp(args, model)
        logger.info("[NANO_BANANA] GCP generation succeeded")
        return result
    except ValueError as e:
        # Check if this is a fallback-eligible error
        if fal_fallback_tool and _should_fallback_to_fal(e):
            logger.warning(
                f"[NANO_BANANA] GCP failed with fallback-eligible error: {e}"
            )
            logger.info(
                f"[NANO_BANANA] >>> FALLING BACK TO FAL <<< using {fal_fallback_tool}"
            )
            result = await _nano_banana_fal_fallback(args, fal_fallback_tool)
            logger.info("[NANO_BANANA] FAL fallback succeeded")
            return result
        logger.error(f"[NANO_BANANA] GCP failed with non-fallback error: {e}")
        raise
